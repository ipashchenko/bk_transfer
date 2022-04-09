import os
from abc import ABC
import functools
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from astropy import units as u, cosmology
import sys
sys.path.insert(0, '/home/ilya/github/ve/vlbi_errors')
from uv_data import UVData, downscale_uvdata_by_freq
from from_fits import create_clean_image_from_fits_file, create_image_from_fits_file, create_model_from_fits_file
from image import plot as iplot
from spydiff import clean_difmap
sys.path.insert(0, '/home/ilya/github/alpha')
from alpha_utils import CLEAN_difmap
try:
    from fourier import FINUFFT_NUNU
except ImportError:
    raise Exception("Install pynufft")
    FINUFFT_NUNU = None
from vlbi_utils import (find_image_std, find_bbox, convert_difmap_model_file_to_CCFITS, rotate_difmap_model)


from matplotlib.colors import LinearSegmentedColormap

colors = [ (0.0,0.00,0.20), (0.00,0.00,0.40) ,(0, 0, 0.8), (0.19,0.88,0.8),  (0.44,0.60,0.74), (0,0.7,0),(1.00,1.00,0.01), (0.9, 0, 0),(1,0,0),(0.80,0.13,0.00), (0.57,0.35,0.35) , (0.55,0.00,1.00), (1,1,1) ]
n_bin = 200  # Discretizes the interpolation into bins
cmap_name = 'my_list'
# Create the colormap
cm_my = LinearSegmentedColormap.from_list(
    cmap_name, colors, N=n_bin)

def rebase_CLEAN_model(target_uvfits, rebased_uvfits, stokes, mapsize, restore_beam, source_ccfits=None,
                       source_difmap_model=None, noise_scale_factor=1.0, need_downscale_uv=None, remove_cc=False):
    if source_ccfits is None and source_difmap_model is None:
        raise Exception("Must specify CCFITS or difmap model file!")
    if source_ccfits is not None and source_difmap_model is not None:
        raise Exception("Must specify CCFITS OR difmap model file!")
    uvdata = UVData(target_uvfits)
    if need_downscale_uv is None:
        need_downscale_uv = downscale_uvdata_by_freq(uvdata)
    noise = uvdata.noise(average_freq=False, use_V=False)
    uvdata.zero_data()
    # If one needs to decrease the noise this is the way to do it
    for baseline, baseline_noise_std in noise.items():
        noise.update({baseline: noise_scale_factor*baseline_noise_std})

    if source_ccfits is not None:
        ccmodel = create_model_from_fits_file(source_ccfits)
    if source_difmap_model is not None:
        convert_difmap_model_file_to_CCFITS(source_difmap_model, stokes, mapsize,
                                            restore_beam, target_uvfits, "tmp_cc.fits")
        ccmodel = create_model_from_fits_file("tmp_cc.fits")
        if remove_cc:
            os.unlink("tmp_cc.fits")

    uvdata.substitute([ccmodel])
    uvdata.noise_add(noise)
    uvdata.save(rebased_uvfits, rewrite=True, downscale_by_freq=need_downscale_uv)


class JetImage(ABC):
    cosmo = cosmology.WMAP9
    """
        ``rot=0`` corresponds to South. ``rot>0`` means rotation clock-wise from
        South.
        ''dx>0`` shifts to negative RA.
        ''dy>0`` shifts to negative DEC.
    """
    def __init__(self, z, n_along, n_across, lg_pixel_size_mas_min, lg_pixel_size_mas_max, ft_class=FINUFFT_NUNU,
                 jet_side=True, dx=0.0, dy=0.0, rot=0.0, stokes="I"):
        self.jet_side = jet_side
        self.z = z
        self.lg_pixel_size_mas_min = lg_pixel_size_mas_min
        self.lg_pixel_size_mas_max = lg_pixel_size_mas_max
        self.n_along = n_along
        self.n_across = n_across

        # resolutions = np.logspace(lg_pixel_size_mas_min, lg_pixel_size_mas_max, n_along)
        resolutions = np.sqrt(np.linspace(1, n_along, n_along))*10**lg_pixel_size_mas_min

        # 2D array of u.angle pixsizes
        self.pixsize_array = np.tile(resolutions, n_across).reshape(n_across, n_along).T*u.mas
        self.ft_class = ft_class
        self.ft_instance = None
        self.calculate_grid()
        self._image = None
        self._image_tau = None
        self.stokes = stokes

        # For compatibility with FT realization and possible future use
        # shift in mas
        self.dx = dx
        self.dy = dy
        # rotation angle in rad
        self.rot = rot

    def plot_resolutions(self):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 1)
        axes.plot(np.cumsum(self.pixsize_array[:, 0]),
                  np.cumsum(self.pixsize_array[:, :int(self.n_across/2)], axis=1)[:, -1])
        axes.set_xlabel("Along distance, mas")
        axes.set_ylabel("Across distance, mas")
        axes.set_aspect("auto")
        return fig

    def halfphi_app_max(self):
        """
        Maximum half opening-angle [rad] that can be imaged with given resolution.
        """
        return np.arctan(np.sum(self.pixsize_array[-1, :int(self.n_across/2)]) / np.sum(self.pixsize_array[:, 0]))

    def calculate_grid(self):
        """
        Calculate grid of ``(r_ob, d)`` - each point is in the center of the
        corresponding pixel. Thus, ``JetModelZoom.ft`` method should not shift
        phases on a half of a pixel. ``r_ob`` and ``d`` are in parsecs.
        """
        pc_x = np.cumsum(self.pix_to_pc, axis=0)-self.pix_to_pc/2
        pc_y_up = self.pix_to_pc[:, :self.pix_to_pc.shape[1]//2][::-1]
        pc_y_low = self.pix_to_pc[:, self.pix_to_pc.shape[1]//2:]
        pc_y_up = (np.cumsum(pc_y_up, axis=1) - pc_y_up/2)[::-1]
        pc_y_low = np.cumsum(pc_y_low, axis=1) - pc_y_low/2
        pc_y = np.hstack((pc_y_up[:, ::-1], -pc_y_low))
        self.r_ob = pc_x
        if not self.jet_side:
            self.r_ob *= -1
        # FIXME: In analogy with older convention we need ``-`` here
        self.d = -pc_y

    @property
    def pc_to_mas(self):
        return (u.pc/self.ang_to_dist).to(u.mas)

    @property
    def r_ob_mas(self):
        return self.r_ob*self.pc_to_mas

    @property
    def d_mas(self):
        return self.d*self.pc_to_mas

    @property
    def imgsize(self):
        return np.max(np.cumsum(self.pixsize_array, axis=0)), \
               np.max(np.cumsum(self.pixsize_array, axis=1))

    @property
    def img_extent(self):
        s0, s1 = self.imgsize[0].value, self.imgsize[1].value
        if self.jet_side:
            return 0, s0, -s1/2, s1/2
        else:
            return -s0, 0, -s1/2, s1/2

    def load_image_stokes(self, stokes, image_stokes_file, scale=1.0):
        self.stokes = stokes
        image = np.loadtxt(image_stokes_file)
        print("Loaded image with Stokes {} and total flux = {} Jy, shape = {}".format(stokes, np.nansum(image), image.shape))
        print("Scaling image x ", scale)
        image *= scale
        image[np.isnan(image)] = 0.0
        self._image = image

    def save_image_to_difmap_format(self, difmap_format_file, scale=1.0):
        with open(difmap_format_file, "w") as fo:
            for idx, imval in np.ndenumerate(self._image):
                if imval == 0:
                    continue
                j, i = idx
                dec = -self.r_ob_mas[i, j].value
                ra = -self.d_mas[i, j].value
                # print("RA = {}, DEC = {}".format(ra, dec))
                fo.write("{} {} {}\n".format(imval*scale, np.hypot(ra, dec), np.rad2deg(np.arctan2(ra, dec))))

    def image(self):
        return self._image

    def image_intensity(self):
        # Factor that accounts non-uniform pixel size in plotting
        factor = (self.pixsize_array/np.min(self.pixsize_array))**2
        return self.image()/factor.T

    def image_tau(self):
        return self._image_tau

    def ft(self, uv, rot=None):
        if rot is None:
            rot = self.rot
        # print("Doing FT of jet model with rot = {}".format(np.rad2deg(rot)))
        mas_to_rad = u.mas.to(u.rad)
        rot = np.array([[np.cos(rot), -np.sin(rot)],
                        [np.sin(rot), np.cos(rot)]])

        # No need in half pixel shifts cause coordinates are already at pixel
        # centers
        shift = [self.dx*mas_to_rad, self.dy*mas_to_rad]
        result = np.exp(-2.0*np.pi*1j*(uv @ shift))
        uv = uv @ rot

        x = (self.d*u.pc/self.ang_to_dist).to(u.rad).value
        y = (self.r_ob*u.pc/self.ang_to_dist).to(u.rad).value
        ft_instance = self.ft_class(uv, x.ravel(), y.ravel())
        img = self.image()
        result *= ft_instance.forward(img.T.ravel())
        del ft_instance, x, y

        return result

    @property
    @functools.lru_cache()
    def ang_to_dist(self):
        return self.cosmo.kpc_proper_per_arcmin(self.z)

    @property
    @functools.lru_cache()
    def pix_to_pc(self):
        """
        2D array of pixel sizes in parsecs.
        """
        return (self.pixsize_array * self.ang_to_dist).to(u.pc).value


class TwinJetImage(object):

    def __init__(self, jet, cjet):
        self.jet = jet
        self.counterjet = cjet
        self.stokes = jet.stokes
        self.rot = self.jet.rot
        assert jet.rot == cjet.rot

    def ft(self, uv, rot=None):
        if rot is None:
            rot = self.rot
        return self.jet.ft(uv, rot) + self.counterjet.ft(uv, rot)

    def substitute_uvfits(self, in_uvfits, out_uvfits, average_freq=False, use_V=False, noise_scale_factor=1.0,
                          need_downscale_uv=None):
        uvdata = UVData(in_uvfits)
        noise = uvdata.noise(average_freq=average_freq, use_V=use_V)
        for baseline, baseline_noise_std in noise.items():
            noise.update({baseline: noise_scale_factor*baseline_noise_std})
        uvdata.zero_data()
        uvdata.substitute([self])
        uvdata.noise_add(noise)
        if need_downscale_uv is None:
            need_downscale_uv = downscale_uvdata_by_freq(uvdata)
        uvdata.save(out_uvfits, rewrite=True, downscale_by_freq=need_downscale_uv)

    def save_image_to_difmap_format(self, difmap_format_file, scale=1.0):
        image = np.hstack((self.counterjet._image[::-1], self.jet._image))
        # CJ has its DEC coordinates multiplied on -1.
        r_ob_mas = np.vstack((self.counterjet.r_ob_mas, self.jet.r_ob_mas))
        d_mas = np.vstack((self.counterjet.d_mas, self.jet.d_mas))

        with open(difmap_format_file, "w") as fo:
            for idx, imval in np.ndenumerate(image):
                if imval == 0:
                    continue
                j, i = idx
                dec = -r_ob_mas[i, j].value
                ra = -d_mas[i, j].value
                # print("RA = {}, DEC = {}".format(ra, dec))
                fo.write("{} {} {}\n".format(imval*scale, np.hypot(ra, dec), np.rad2deg(np.arctan2(ra, dec))))


if __name__ == "__main__":
    save_dir = "/home/ilya/data/alpha/LeshaPaper/alpha_kh"
    txt_box = os.path.join(save_dir, "wins.txt")
    make_true_model_convolved = True
    rot_angle_deg = -110.
    freqs_ghz = (8.4, 15.3)
    bands = ("X", "U")
    common_mapsize = (1024, 0.1)
    uvrange = [1.287, 242.1]
    noise_scale_factor = 1.0
    z = 0.00436
    # n_along = 1000
    n_along = 350
    # n_across = 200
    n_across = 100
    # lg_pixel_size_mas_min = -2.5
    lg_pixel_size_mas_min = -2.0
    lg_pixel_size_mas_max = -0.5
    freq_bands_dict = {"X": 8.4, "U": 15.3}
    # imsize 2048
    # blc = (900, 940)
    # trc = (1700, 1300)
    # imsize 1024
    blc = (450, 450)
    trc = (980, 720)
    color_clim = [-1.5, 0.5]
    template_ccimage = {"X": "/home/ilya/data/alpha/MOJAVE/template_cc_i_8.1.fits",
                        "U": "/home/ilya/data/alpha/MOJAVE/template_cc_i_15.4.fits"}
    template_ccimage = create_clean_image_from_fits_file(template_ccimage["X"])
    common_beam = template_ccimage.beam
    small_beam = (0.5, 0.5, 0)
    npixels_beam_common = np.pi*common_beam[0]*common_beam[1]/(4*np.log(2)*common_mapsize[1]**2)
    template_uvfits_dict = {"X": "/home/ilya/data/alpha/BK145/1228+126.X.2009_05_23_ta60.uvf_cal",
                            "U": "/home/ilya/data/alpha/BK145/1228+126.U.2009_05_23C_ta60.uvf_cal"}

    art_uvfits_dict = {"X": os.path.join(save_dir, "BK145_U_artificial_KH.uvf"),
                       "U": os.path.join(save_dir, "BK145_X_artificial_KH.uvf")}
    art_ccfits_dict = {"U": os.path.join(save_dir, "BK145_U_artificial_KH_cc.fits"),
                       "X": os.path.join(save_dir, "BK145_X_artificial_KH_cc.fits")}
    art_rebased_uvfits = os.path.join(save_dir, "BK145_U2X_artificial_KH.uvf")
    art_rebased_ccfits = os.path.join(save_dir, "BK145_U2X_artificial_KH_cc.fits")
    art_rebased_dccfits = os.path.join(save_dir, "BK145_U2X_artificial_KH_dcc.fits")
    spix_picture_orig = os.path.join(save_dir, "BK145_spix_orig_artificial_KH.png")
    spix_picture_cor = os.path.join(save_dir, "BK145_spix_cor_artificial_KH.png")
    path_to_script = "/home/ilya/github/bk_transfer/scripts/final_clean_nw"
    # To quickly show J + CJ images for freq_ghz GHz:
    # j = np.loadtxt("jet_image_i_{}.txt".format(freq_ghz)); cj = np.loadtxt("cjet_image_i_{}.txt".format(freq_ghz)); jcj = np.hstack((cj[::, ::-1], j)); plt.matshow(jcj, aspect="auto");plt.colorbar(); plt.show()

    # Create X & U artificial data sets
    for band in bands:
        uvdata = UVData(template_uvfits_dict[band])
        noise = uvdata.noise(average_freq=False, use_V=False)
        uvdata.zero_data()
        # If one needs to decrease the noise this is the way to do it
        for baseline, baseline_noise_std in noise.items():
            noise.update({baseline: noise_scale_factor*baseline_noise_std})

        jm = JetImage(z=z, n_along=n_along, n_across=n_across,
                      lg_pixel_size_mas_min=lg_pixel_size_mas_min, lg_pixel_size_mas_max=lg_pixel_size_mas_max,
                      jet_side=True, rot=np.deg2rad(rot_angle_deg))
        cjm = JetImage(z=z, n_along=n_along, n_across=n_across,
                       lg_pixel_size_mas_min=lg_pixel_size_mas_min, lg_pixel_size_mas_max=lg_pixel_size_mas_max,
                       jet_side=False, rot=np.deg2rad(rot_angle_deg))
        jm.load_image_stokes("I", "/home/ilya/github/bk_transfer/Release/jet_image_i_{}.txt".format(freq_bands_dict[band]), scale=1.0)
        cjm.load_image_stokes("I", "/home/ilya/github/bk_transfer/Release/cjet_image_i_{}.txt".format(freq_bands_dict[band]), scale=1.0)
        js = TwinJetImage(jm, cjm)

        js.save_image_to_difmap_format("{}/true_jet_model_i_{}.txt".format(save_dir, band))
        js.substitute_uvfits(template_uvfits_dict[band], art_uvfits_dict[band], need_downscale_uv=True)

        # Rotate
        rotate_difmap_model("{}/true_jet_model_i_{}.txt".format(save_dir, band),
                            "{}/true_jet_model_i_{}_rotated.txt".format(save_dir, band),
                            PA_deg=-rot_angle_deg)
        if make_true_model_convolved:
            # Convolve with beam
            print("Convolving true I at {} with a common beam {}".format(band, common_beam))
            convert_difmap_model_file_to_CCFITS("{}/true_jet_model_i_{}_rotated.txt".format(save_dir, band), "I", common_mapsize,
                                                common_beam, template_uvfits_dict[band],
                                                "{}/convolved_true_jet_model_i_rotated_{}.fits".format(save_dir, band))
            print("Convolving true I at {} with a small beam beam {}".format(band, small_beam))
            convert_difmap_model_file_to_CCFITS("{}/true_jet_model_i_{}_rotated.txt".format(save_dir, band), "I", common_mapsize,
                                                small_beam, template_uvfits_dict[band],
                                                "{}/convolved_true_jet_model_i_rotated_{}_small_beam.fits".format(save_dir, band))
        if band == "U":
            dmap = art_rebased_dccfits
        else:
            dmap = None
        clean_difmap(fname=art_uvfits_dict[band], outfname=art_ccfits_dict[band], stokes="I",
                     mapsize_clean=common_mapsize, path_to_script=path_to_script, show_difmap_output=True,
                     text_box=None, beam_restore=common_beam, dmap=dmap)
        # CLEAN_difmap(uvfits=art_uvfits_dict[band], stokes="I", mapsize=common_mapsize, outname=art_ccfits_dict[band], restore_beam=common_beam,
        #              boxfile=txt_box, working_dir=save_dir, uvrange=None,
        #              box_clean_nw_niter=1000, clean_gain=0.03, dynam_su=20, dynam_u=6, deep_factor=3.0,
        #              remove_difmap_logs=True, save_noresid=None, save_resid_only=dmap, save_dfm=None,
        #              noise_to_use="F")

    # Now re-base 15 GHz model on 8 GHz uv
    rebase_CLEAN_model(art_uvfits_dict["X"], art_rebased_uvfits, "I", common_mapsize, common_beam,
                       source_ccfits=art_ccfits_dict["U"], noise_scale_factor=0.1, need_downscale_uv=True)

    # CLEAN this re-based data set
    clean_difmap(fname=art_rebased_uvfits, outfname=art_rebased_ccfits, stokes="I", mapsize_clean=common_mapsize,
                 path_to_script=path_to_script, show_difmap_output=True, text_box=None, beam_restore=common_beam)
    # CLEAN_difmap(uvfits=art_rebased_uvfits, stokes="I", mapsize=common_mapsize, outname=art_rebased_ccfits,
    #              restore_beam=common_beam, boxfile=txt_box, working_dir=save_dir, uvrange=None,
    #              box_clean_nw_niter=1000, clean_gain=0.03, dynam_su=20, dynam_u=6, deep_factor=3.0,
    #              remove_difmap_logs=True, save_noresid=None, save_resid_only=None, save_dfm=None,
    #              noise_to_use="F")

    # True convolved SPIX
    itrue_convolved_low = create_image_from_fits_file("{}/convolved_true_jet_model_i_rotated_{}.fits".format(save_dir, "X")).image
    itrue_convolved_high = create_image_from_fits_file("{}/convolved_true_jet_model_i_rotated_{}.fits".format(save_dir, "U")).image
    true_convolved_spix_array = np.log(itrue_convolved_low/itrue_convolved_high)/np.log(freq_bands_dict["X"]/freq_bands_dict["U"])
    # True convolved SPIX - small beam
    itrue_convolved_low_sb = create_image_from_fits_file("{}/convolved_true_jet_model_i_rotated_{}_small_beam.fits".format(save_dir, "X")).image
    itrue_convolved_high_sb = create_image_from_fits_file("{}/convolved_true_jet_model_i_rotated_{}_small_beam.fits".format(save_dir, "U")).image
    true_convolved_spix_array_sb = np.log(itrue_convolved_low_sb/itrue_convolved_high_sb)/np.log(freq_bands_dict["X"]/freq_bands_dict["U"])

    # Obtained original artificial SPIX map
    ccimages_orig = {band: create_clean_image_from_fits_file(art_ccfits_dict[band]) for band in bands}
    dimage = create_image_from_fits_file(art_rebased_dccfits)
    ccimage_rebased = create_clean_image_from_fits_file(art_rebased_ccfits)

    # Find std and common mask
    ipol_arrays = dict()
    masks_dict = dict()
    std_dict = dict()
    for band in bands:
        ipol = ccimages_orig[band].image
        ipol_arrays[band] = ipol
        std = find_image_std(ipol, beam_npixels=npixels_beam_common)
        std_dict[band] = std
        masks_dict[band] = ipol < 4*std

    common_imask = np.logical_or.reduce([masks_dict[band] for band in bands])

    ipol_array_rebased = ccimage_rebased.image + dimage.image

    spix_array = np.log(ipol_arrays["X"]/ipol_arrays["U"])/np.log(freq_bands_dict["X"]/freq_bands_dict["U"])
    spix_array_corr = np.log(ipol_arrays["X"]/ipol_array_rebased)/np.log(freq_bands_dict["X"]/freq_bands_dict["U"])

    fig = iplot(ipol_arrays["X"], spix_array, x=ccimages_orig["X"].x, y=ccimages_orig["X"].y,
                min_abs_level=4*std_dict["X"], colors_mask=common_imask, color_clim=color_clim, blc=blc, trc=trc,
                beam=common_beam, close=True, colorbar_label=r"$\alpha_{\rm orig}$", show_beam=True, show=False,
                cmap=cm_my, contour_color='black', plot_colorbar=True, contour_linewidth=0.25,
                beam_place="lr")
    fig.savefig(spix_picture_orig, dpi=600, bbox_inches="tight")

    fig = iplot(ipol_arrays["X"], spix_array_corr, x=ccimages_orig["X"].x, y=ccimages_orig["X"].y,
                min_abs_level=4*std_dict["X"], colors_mask=common_imask, color_clim=color_clim, blc=blc, trc=trc,
                beam=common_beam, close=True, colorbar_label=r"$\alpha_{\rm cor}$", show_beam=True, show=False,
                cmap=cm_my, contour_color='black', plot_colorbar=True, contour_linewidth=0.25,
                beam_place="lr")
    fig.savefig(spix_picture_cor, dpi=600, bbox_inches="tight")

    fig = iplot(itrue_convolved_high_sb, true_convolved_spix_array_sb, x=ccimages_orig["X"].x, y=ccimages_orig["X"].y,
                min_abs_level=0.0001*np.max(itrue_convolved_high_sb), colors_mask=itrue_convolved_high_sb < 0.0001*np.max(itrue_convolved_high_sb),
                color_clim=color_clim, blc=blc, trc=trc,
                beam=small_beam, close=True, colorbar_label=r"$\alpha_{\rm true\,conv}$", show_beam=True, show=False,
                cmap=cm_my, contour_color='black', plot_colorbar=True,
                contour_linewidth=0.25, beam_place="lr")
    fig.savefig(os.path.join(save_dir, "true_conv_spix_UX_true_I_U_small_beam.png"), dpi=600, bbox_inches="tight")

    fig = iplot(itrue_convolved_low_sb, true_convolved_spix_array_sb, x=ccimages_orig["X"].x, y=ccimages_orig["X"].y,
                min_abs_level=0.0001*np.max(itrue_convolved_low_sb), colors_mask=itrue_convolved_low_sb < 0.0001*np.max(itrue_convolved_low_sb),
                color_clim=color_clim, blc=blc, trc=trc,
                beam=small_beam, close=True, colorbar_label=r"$\alpha_{\rm true\,conv}$", show_beam=True, show=False,
                cmap=cm_my, contour_color='black', plot_colorbar=True,
                contour_linewidth=0.25, beam_place="lr")
    fig.savefig(os.path.join(save_dir, "true_conv_spix_UX_true_I_X_small_beam.png"), dpi=600, bbox_inches="tight")

    fig = iplot(ipol_arrays["X"], spix_array - true_convolved_spix_array, x=ccimages_orig["X"].x, y=ccimages_orig["X"].y,
                min_abs_level=4*std_dict["X"], colors_mask=common_imask, color_clim=[-0.2, 0.2], blc=blc, trc=trc,
                beam=common_beam, close=True, colorbar_label=r"$\alpha_{\rm orig}$ bias", show_beam=True, show=False,
                cmap='bwr', contour_color='black', plot_colorbar=True,
                contour_linewidth=0.25, beam_place="lr")
    fig.savefig(os.path.join(save_dir, "bias_orig_spix_UX_I_X.png"), dpi=600, bbox_inches="tight")

    fig = iplot(ipol_arrays["X"], spix_array_corr - true_convolved_spix_array, x=ccimages_orig["X"].x, y=ccimages_orig["X"].y,
                min_abs_level=4*std_dict["X"], colors_mask=common_imask, color_clim=[-0.2, 0.2], blc=blc, trc=trc,
                beam=common_beam, close=True, colorbar_label=r"$\alpha_{\rm cor}$ bias", show_beam=True, show=False,
                cmap='bwr', contour_color='black', plot_colorbar=True,
                contour_linewidth=0.25, beam_place="lr")
    fig.savefig(os.path.join(save_dir, "bias_cor_spix_UX_I_X.png"), dpi=600, bbox_inches="tight")