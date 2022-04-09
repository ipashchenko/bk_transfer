import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import astropy.io.fits as pf
import astropy.units as un
from astropy.stats import gaussian_fwhm_to_sigma
from jet_image import JetImage, TwinJetImage
from vlbi_utils import (find_image_std, convert_difmap_model_file_to_CCFITS, rotate_difmap_model, get_uvrange,
                        find_bbox, FWHM_ell_beam_slice)
import sys
sys.path.insert(0, '/home/ilya/github/ve/vlbi_errors')
from uv_data import UVData, downscale_uvdata_by_freq
from from_fits import create_image_from_fits_file
from image import plot as iplot
sys.path.insert(0, '/home/ilya/github/alpha')
from alpha_utils import CLEAN_difmap_RA



deg2rad = un.deg.to(un.rad)


done_convolved = False
done_artificial = False

# freq = 4.8
freq = 15.4
# MOJAVE:  (8.950327, 432.140576)
# RA:  (1.82467625, 652.150144)
# Excluding outer parts of RA uv-coverage
uvrange_ra = [8.950327, 300]
# Trivial
uvrange = [0, 3000]

jet_model = "2ridges"
# FIXME: What about common beam for RA - I think it was 0.8 mas!!!
common_beam_deg = (0.7, 0.7, 0)
common_beam_rad = (0.7, 0.7, np.deg2rad(0))
common_mapsize = (2048, 1024, 0.04)
data_dir = "/home/ilya/data/alpha/RA"
save_dir = "/home/ilya/data/alpha/RA"
uvfits_ra = "/home/ilya/data/alpha/RA/m87.at02.uvf_cal"
uvfits_moj = "/home/ilya/data/alpha/RA/1228+126.u.2014_03_26.uvf"
wins_file_ra = "/home/ilya/data/alpha/RA/wins_RA_2ridges.txt"
noise_scale_factor = 1.0
stokes = "I"
scale = 1.0


rot_angle_deg = -107.0
jetpol_files_directory = "/home/ilya/github/bk_transfer/Release"
z = 0.00436
n_along = 1400
n_across = 500
lg_pixel_size_mas_min = -1.5
lg_pixel_size_mas_max = -1.5


if not done_artificial:
    # uvdata = UVData(uvfits_ra)
    uvdata = UVData(uvfits_moj)
    # RA
    # need_downscale_uv = True
    # MOJAVE
    need_downscale_uv = False
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
    jm.load_image_stokes(stokes.upper(), "{}/jet_image_{}_{}_{}.txt".format(jetpol_files_directory, "i", freq, jet_model), scale=scale)
    cjm.load_image_stokes(stokes.upper(), "{}/cjet_image_{}_{}_{}.txt".format(jetpol_files_directory, "i", freq, jet_model), scale=scale)
    js = TwinJetImage(jm, cjm)
    # Convert to difmap model format
    js.save_image_to_difmap_format("{}/true_jet_model_i_{}.txt".format(save_dir, freq))
    # Rotate
    rotate_difmap_model("{}/true_jet_model_i_{}.txt".format(save_dir, freq),
                        "{}/true_jet_model_i_{}_rotated.txt".format(save_dir, freq),
                        PA_deg=-rot_angle_deg)

    if not done_convolved:
        # Convolve with beam
        convert_difmap_model_file_to_CCFITS("{}/true_jet_model_i_{}_rotated.txt".format(save_dir, freq), "i", common_mapsize,
                                            common_beam_rad, uvfits_ra,
                                            "{}/convolved_true_jet_model_i_rotated_{}.fits".format(save_dir, freq))
    uvdata.substitute([js])
    uvdata.noise_add(noise)
    uvdata.save(os.path.join(save_dir, "template_{}.uvf".format(freq)), rewrite=True, downscale_by_freq=need_downscale_uv)


sys.exit(0)

# No need uv-clipping from the above.
CLEAN_difmap_RA("template_{}.uvf".format(freq), "ll", common_mapsize, "model_cc_i_{}".format(freq), restore_beam=common_beam_deg,
                boxfile=wins_file_ra, working_dir=save_dir,
                # uvrange=uvrange_ra,
                box_clean_nw_niter=1000, clean_gain=0.03, dynam_su=20, dynam_u=6, deep_factor=1.0,
                remove_difmap_logs=False, save_noresid=True, save_resid_only=True, save_dfm=True,
                noise_to_use="F")


uvfits = os.path.join(save_dir, "template_{}.uvf".format(freq))
model_dfm = os.path.join(save_dir, "model_cc_i_{}.dfm".format(freq))
noresid_ccfits = os.path.join(save_dir, "model_cc_i_{}_noresid.fits".format(freq))
resid_ccfits = os.path.join(save_dir, "model_cc_i_{}_resid_only.fits".format(freq))
ccfits = os.path.join(save_dir, "model_cc_i_{}.fits".format(freq))
model_fits = os.path.join(save_dir, "convolved_true_jet_model_i_rotated_{}.fits".format(freq))


uvdata = UVData(os.path.join(data_dir, uvfits))
uv = uvdata.uv.copy()
u = uv[:, 0]
v = uv[:, 1]
r_uv = np.hypot(uv[:, 0], uv[:, 1])
pa_uv = np.arctan2(u, v)
ll = uvdata.uvdata_freq_averaged[:, 1]
weights_mask = ll.mask

cc_data = pf.getdata(os.path.join(data_dir, noresid_ccfits), extname="AIPS CC")
model_data = pf.getdata(os.path.join(data_dir, model_fits), extname="AIPS CC")

components_cc = list()
for flux, x, y in zip(cc_data['FLUX'], cc_data['DELTAX']*deg2rad,
                      cc_data['DELTAY']*deg2rad):
    # We keep positions in mas
    components_cc.append((flux, -x, -y))

components_model = list()
for flux, x, y in zip(model_data['FLUX'], model_data['DELTAX']*deg2rad,
                      model_data['DELTAY']*deg2rad):
    # We keep positions in mas
    components_model.append((flux, -x, -y))

ft_cc = np.zeros(len(uv), dtype=complex)
ft_model = np.zeros(len(uv), dtype=complex)

for flux, x0, y0 in components_cc:
    ft_cc += (flux*np.exp(-2.0*np.pi*1j*(u[:, np.newaxis]*x0 +
                                         v[:, np.newaxis]*y0))).sum(axis=1)

for flux, x0, y0 in components_model:
    ft_model += (flux*np.exp(-2.0*np.pi*1j*(u[:, np.newaxis]*x0 +
                                            v[:, np.newaxis]*y0))).sum(axis=1)

good = np.logical_and(r_uv/10**6 > uvrange[0], r_uv/10**6 < uvrange[1])
toogood = np.logical_and(good, ~weights_mask)


fwhm_beam_mas = common_beam_deg[0]
fwhm_beam_rad = fwhm_beam_mas*un.mas.to(un.rad)
# sigma_beam_rad = fwhm_beam_rad/np.sqrt(2*np.log(2))
# sigma_beam_uv = 1/(2*np.pi*sigma_beam_rad)
fwhm_uv = 4*np.log(2)/np.pi/fwhm_beam_rad
sigma_uv = gaussian_fwhm_to_sigma*fwhm_uv
print("FWHM (0.7mas) = ", fwhm_uv/10**6)
print("sigma (0.7mas) = ", sigma_uv/10**6)


def factor(u, v, sigma_uv):
    return np.exp(-(u**2 + v**2)/(2*sigma_uv**2))


fig, axes = plt.subplots(1, 1, figsize=[6.4, 4.8])
y_cc = np.real(np.sqrt(ft_cc*np.conj(ft_cc)))
y_model = np.real(np.sqrt(ft_model*np.conj(ft_model)))
diff = ft_cc - ft_model
diff = np.real(np.sqrt(diff*np.conj(diff)))
sc = axes.scatter(r_uv[toogood]/10**6, diff[toogood]/y_model[toogood], c=np.rad2deg(pa_uv[toogood]), cmap="plasma")
# sc = axes.scatter(r_uv[toogood]/10**6, diff[toogood], c=np.rad2deg(pa_uv[toogood]), cmap="plasma")
axes.set_yscale("log")
axes.set_xlabel(r"$r_{uv}$, M$\lambda$")
axes.set_ylabel(r"Frac. error")
# axes.set_ylabel(r"Error, Jy")
divider = make_axes_locatable(axes)
cax = divider.append_axes("right", size="10%", pad=0.00)
cb = fig.colorbar(sc, cax=cax)
cb.set_label(r"PA$_{uv}$, deg")
# plt.savefig(os.path.join(data_dir, "radplot_error.png"), bbox_inches="tight", dpi=600)
plt.show()


fig, axes = plt.subplots(1, 1, figsize=[10, 5])
# sc = axes.scatter(u[toogood]/10**6, v[toogood]/10**6, c=np.log10(diff[toogood]/y_model[toogood]), cmap="plasma")
# axes.scatter(-u[toogood]/10**6, -v[toogood]/10**6, c=np.log10(diff[toogood]/y_model[toogood]), cmap="plasma")
sc = axes.scatter(u[toogood]/10**6, v[toogood]/10**6, c=np.log10(diff[toogood]), vmin=-3.5, cmap="plasma")
axes.scatter(-u[toogood]/10**6, -v[toogood]/10**6, c=np.log10(diff[toogood]), vmin=-3.5, cmap="plasma")
divider = make_axes_locatable(axes)
cax = divider.append_axes("right", size="3%", pad=0.00)
cb = fig.colorbar(sc, cax=cax)
# cb.set_label(r"$\lg({\rmFrac. error})$")
cb.set_label(r"$\lg({\rm Error (Jy)})$")
axes.set_xlabel(r"$u$, M$\lambda$")
axes.set_ylabel(r"$v$, M$\lambda$")
axes.set_aspect("equal")
# axes.set_ylim([-120, 120])
# axes.set_ylim([-210, 210])
# axes.set_xlim([-210, 210])
axes.invert_xaxis()
plt.savefig(os.path.join(save_dir, "uv_abs_error_CLEAN_reconstruction.png"), bbox_inches="tight", dpi=600)
plt.show()


# Bias image
i_true = create_image_from_fits_file(model_fits)
i_obs = create_image_from_fits_file(ccfits)
i_resid = create_image_from_fits_file(resid_ccfits)
bias = i_obs.image - i_true.image
npixels_beam = np.pi*common_beam_rad[0]*common_beam_rad[1]/(4*np.log(2)*common_mapsize[1]**2)
std = find_image_std(i_obs.image, beam_npixels=npixels_beam)
blc, trc = find_bbox(i_obs.image, level=3*std, min_maxintensity_mjyperbeam=10*std, min_area_pix=10*npixels_beam, delta=10)

fig = iplot(i_obs.image, bias/i_true.image, x=i_obs.x, y=i_obs.y,
            min_abs_level=3*std, colors_mask=i_obs.image < 3*std, color_clim=[-0.5, 0.5], blc=blc, trc=trc,
            beam=common_beam_rad, close=True, colorbar_label=r"$I$ frac. bias", show_beam=True, show=False,
            cmap='bwr', contour_color='black', plot_colorbar=True,
            contour_linewidth=0.25, beam_place="lr")
fig.savefig(os.path.join(save_dir, "frac_bias_I.png"), dpi=600, bbox_inches="tight")

fig = iplot(i_obs.image, i_resid.image/i_true.image, x=i_obs.x, y=i_obs.y,
            min_abs_level=3*std, colors_mask=i_obs.image < 3*std, color_clim=[-0.5, 0.5], blc=blc, trc=trc,
            beam=common_beam_rad, close=True, colorbar_label=r"$I$ frac. bias", show_beam=True, show=False,
            cmap='bwr', contour_color='black', plot_colorbar=True,
            contour_linewidth=0.25, beam_place="lr")
fig.savefig(os.path.join(save_dir, "frac_residuals_I.png"), dpi=600, bbox_inches="tight")
