import os
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import numpy as np
import matplotlib
# matplotlib.use("TkAgg")
import scienceplots
import matplotlib.pyplot as plt
import scienceplots
import matplotlib.pyplot as plt
# For tics and line widths. Re-write colors and figure size later.
try:
    plt.style.use(['science'])
except OSError:
    pass
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cycler import cycler
from vlbi_utils import find_image_std, find_bbox, downscale_uvdata_by_freq
from generate_and_model_many_epochs_uvdata import find_iqu_image_std, pol_mask
from jet_image import JetImage, convert_difmap_model_file_to_CCFITS
sys.path.insert(0, '/home/ilya/github/ve/vlbi_errors')
from uv_data import UVData, get_uvrange
from spydiff import clean_difmap, find_nw_beam
from from_fits import create_clean_image_from_fits_file, create_image_from_fits_file
from image import plot as iplot
from plotting import plot as pplot
from image_ops import spix_map, rotm_map, hovatta_find_sigma_pang
# Default color scheme
matplotlib.rcParams['axes.prop_cycle'] = cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
# Default figure size
matplotlib.rcParams['figure.figsize'] = (6.4, 4.8)

label_size = 18
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size
matplotlib.rcParams['axes.titlesize'] = label_size
matplotlib.rcParams['axes.labelsize'] = label_size
matplotlib.rcParams['font.size'] = label_size
matplotlib.rcParams['legend.fontsize'] = label_size
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42



do_not_reclean = False
jetpol_run_directory = "/home/ilya/github/bk_transfer/cmake-build-debug"
# cwd = os.getcwd()
# os.chdir(jetpol_run_directory)
# os.system("./bk_transfer 0.1, 90, 0.7, 0.1, 0., 1.1, 200, 100, -2, -2")
# os.chdir(cwd)
save_dir = "/home/ilya/data/DDS"
tau_fr_file = "jet_image_taufr_u.txt"
stokes = ("i", "q", "u")

#          ╭──────────────────────────────────────────────────────────╮
#          │                Plotting raw model images                 │
#          ╰──────────────────────────────────────────────────────────╯
tau_fr_image = np.loadtxt(os.path.join(jetpol_run_directory, tau_fr_file))
# FIXME: lambda squared, RM in rad/m^2
rm_image = 0.5*tau_fr_image/(0.02**2)
fig, axes = plt.subplots(1, 1)
im = axes.matshow(tau_fr_image, cmap="bwr")
divider = make_axes_locatable(axes)
cax = divider.append_axes("right", size="5%", pad=0.00)
cb = fig.colorbar(im, cax=cax)
cb.set_label("Faraday depth")
axes.set_xlabel("along")
axes.set_ylabel("across")
fig.savefig(os.path.join(save_dir, "Faraday_depth_image.png"), bbox_inches="tight")
plt.show()


fig, axes = plt.subplots(1, 1)
im = axes.matshow(rm_image, cmap="bwr")
divider = make_axes_locatable(axes)
cax = divider.append_axes("right", size="5%", pad=0.00)
cb = fig.colorbar(im, cax=cax)
cb.set_label(r"RM, rad/m$^2$")
axes.set_xlabel("along")
axes.set_ylabel("across")
fig.savefig(os.path.join(save_dir, "RM_image.png"), bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(1, 1)
axes.plot(tau_fr_image[:, 100])
axes.set_xlabel("across")
axes.set_ylabel("Faraday depth")
axes.set_xlim([0, 100])
fig.savefig(os.path.join(save_dir, "Faraday_depth_slice.png"), bbox_inches="tight")
plt.show()

#          ╭──────────────────────────────────────────────────────────╮
#          │                   Plotting VLBI images                   │
#          ╰──────────────────────────────────────────────────────────╯
redshift = 0.1
noise_scale_factor = 1.0
lg_pixsize_min = -2.
lg_pixsize_max = -2.
n_along = 200
n_across = 100
rot_angle_deg = 0.0
freqs_ghz = [15.4, 12.1, 8.4, 8.1]
freq_names = {15.4: "u", 12.1: "j", 8.4: "y", 8.1: "x"}
names_freq = {i:k for (k, i) in freq_names.items()}
mapsizes_dict = {8.1: (1024, 0.1),
                 8.4: (1024, 0.1),
                 12.1: (1024, 0.1),
                 15.4: (1024, 0.1)}
common_mapsize = (1024, 0.1)
common_mapsize_x2 = (int(common_mapsize[0]*2), common_mapsize[1])
path_to_script = "/home/ilya/github/bk_transfer/scripts/final_clean_nw"
# Some template UVFITS with full polarization. Its uv-coverage and noise will be used while creating fake data
template_uvfits_dict = {"u": "1458+718.u.2006_09_06.uvf", "j": "1458+718.j.2006_09_06.uvf",
                        "y": "1458+718.y.2006_09_06.uvf", "x": "1458+718.x.2006_09_06.uvf"}
template_uvfits_dir = "/home/ilya/github/bk_transfer/uvfits"

# Find common uv-range
uvrange_x = get_uvrange(os.path.join(template_uvfits_dir, template_uvfits_dict["x"]))
uvrange_u = get_uvrange(os.path.join(template_uvfits_dir, template_uvfits_dict["u"]))
print(f"uv-range x : {uvrange_x}, uv-range u : {uvrange_u}")
common_uvrange = (uvrange_u[0], uvrange_x[1])
print(f"Common uv-range : {common_uvrange}")
pass
# Find common beam
nw_beam_x = find_nw_beam(os.path.join(template_uvfits_dir,
                                      template_uvfits_dict["x"]), stokes="I",
                         mapsize=(1024, 0.1), uv_range=common_uvrange, working_dir=None)
common_beam = np.sqrt(nw_beam_x[0]*nw_beam_x[1])
common_beam = (common_beam, common_beam, 0)
# Common beam in (mas, mas, deg). Can be arbitrary small/big
# common_beam = (1, 1, 0)



#          ╭──────────────────────────────────────────────────────────╮
#          │        Plot of model images at highest frequency         │
#          ╰──────────────────────────────────────────────────────────╯
freq_ghz = np.max(freqs_ghz)
i_image = np.loadtxt(os.path.join(jetpol_run_directory, 'jet_image_i_{}.txt'.format(freq_names[freq_ghz])))
q_image = np.loadtxt(os.path.join(jetpol_run_directory, 'jet_image_q_{}.txt'.format(freq_names[freq_ghz])))
u_image = np.loadtxt(os.path.join(jetpol_run_directory, 'jet_image_u_{}.txt'.format(freq_names[freq_ghz])))
v_image = np.loadtxt(os.path.join(jetpol_run_directory, 'jet_image_v_{}.txt'.format(freq_names[freq_ghz])))
tau_image = np.loadtxt(os.path.join(jetpol_run_directory, 'jet_image_tau_{}.txt'.format(freq_names[freq_ghz])))
tau_fr_image = np.loadtxt(os.path.join(jetpol_run_directory, 'jet_image_taufr_{}.txt'.format(freq_names[freq_ghz])))
rm_image = 0.5*tau_fr_image/(0.02**2)
l_image = np.loadtxt(os.path.join(jetpol_run_directory, 'jet_image_l_{}.txt'.format(freq_names[freq_ghz])))
p_image = np.sqrt(q_image**2 + u_image**2)
fpol_image = p_image/i_image
# alpha_image = np.log(i_image_low/i_image_high)/np.log(freq_ghz_low/freq_ghz_high)
chi_image = 0.5*np.arctan2(u_image, q_image) - 0.5*tau_fr_image

# Just plotting picture
colors_mask = i_image < i_image.max()*0.000001
fig = pplot(contours=i_image, colors=fpol_image, vectors=chi_image,
     vectors_values=None, colors_mask=colors_mask, min_rel_level=0.01,
     vinc=10, vectors_mask=colors_mask, vector_color="k", contour_color="k", cmap="jet", color_clim=[0, 0.75],
     colorbar_label=r'$FPOL$', vector_enlarge_factor=8)#, outdir="/home/ilya", outfile="oldMHDsimulations_LinPol_NUpixel.png")
fig.savefig(os.path.join(save_dir, "FPOL_chi_Icontours.png"), dpi=300, bbox_inches="tight")
plt.close()

fig = pplot(contours=p_image, colors=1000*p_image, vectors=chi_image,
           vectors_values=None, colors_mask=colors_mask, min_rel_level=0.01,
           vinc=10, vectors_mask=colors_mask, contour_color="k", vector_color="k", cmap="jet",
           vector_enlarge_factor=8, colorbar_label=r"$P$, mJy/pix")#, outdir="/home/ilya", outfile="oldMHDsimulations_LinPol_NUpixel.png")
fig.savefig(os.path.join(save_dir, "P_chi_Pcontours.png"), dpi=300, bbox_inches="tight")
plt.close()

fig = pplot(contours=i_image, colors=1000*p_image, cmap="jet", colors_mask=colors_mask, min_rel_level=0.05,
           colorbar_label=r'$P$, mJy/pix')
fig.savefig(os.path.join(save_dir, "P_Icontours.png"), dpi=300, bbox_inches="tight")
plt.close()

fig = pplot(contours=p_image, colors=fpol_image, colors_mask=colors_mask, min_rel_level=0.05, color_clim=[0, 0.75],
     colorbar_label=r'$FPOL$', cmap="jet")
fig.savefig(os.path.join(save_dir, "FPOL_Pcontours.png"), dpi=300, bbox_inches="tight")
plt.close()

max_tau_fr = np.max(np.abs(tau_fr_image))
fig = pplot(contours=i_image, colors=tau_fr_image, colors_mask=colors_mask, min_rel_level=0.05,
           colorbar_label=r'$\tau_{\rm FR}$', cmap="bwr", color_clim=[-max_tau_fr, max_tau_fr])#, outdir="/home/ilya", outfile="oldMHDsimulations_LinPol_NUpixel.png")
fig.savefig(os.path.join(save_dir, "FaradayDepth_15GHz_Icontours.png"), dpi=300, bbox_inches="tight")
plt.close()

max_rm = np.max(np.abs(rm_image))
fig = pplot(contours=i_image, colors=rm_image, colors_mask=colors_mask, min_rel_level=0.05,
           colorbar_label=r'RM, rad/m$^2$', color_clim=[-max_rm, max_rm], cmap="bwr")
fig.savefig(os.path.join(save_dir, "RM_Icontours.png"), dpi=300, bbox_inches="tight")
plt.close()



for freq_ghz in freqs_ghz:

#          ╭──────────────────────────────────────────────────────────╮
#          │                 Create synthetic uv-data                 │
#          ╰──────────────────────────────────────────────────────────╯

    nw_beam_size = None

    template_uvfits = os.path.join(template_uvfits_dir, template_uvfits_dict[freq_names[freq_ghz]])
    if nw_beam_size is None:
        nw_beam = find_nw_beam(template_uvfits, "i", mapsize=mapsizes_dict[freq_ghz])
        nw_beam_size = np.sqrt(nw_beam[0]*nw_beam[1])
        print("NW beam size = ", nw_beam_size)
    uvdata = UVData(template_uvfits)
    noise = uvdata.noise(average_freq=False, use_V=False)
    # If one needs to decrease the noise this is the way to do it
    for baseline, baseline_noise_std in noise.items():
        noise.update({baseline: noise_scale_factor*baseline_noise_std})

    jm_i = JetImage(z=redshift, n_along=n_along, n_across=n_across,
                    lg_pixel_size_mas_min=lg_pixsize_min, lg_pixel_size_mas_max=lg_pixsize_max,
                    jet_side=True, rot=np.deg2rad(rot_angle_deg))
    jm_q = JetImage(z=redshift, n_along=n_along, n_across=n_across,
                    lg_pixel_size_mas_min=lg_pixsize_min, lg_pixel_size_mas_max=lg_pixsize_max,
                    jet_side=True, rot=np.deg2rad(rot_angle_deg))
    jm_u = JetImage(z=redshift, n_along=n_along, n_across=n_across,
                    lg_pixel_size_mas_min=lg_pixsize_min, lg_pixel_size_mas_max=lg_pixsize_max,
                    jet_side=True, rot=np.deg2rad(rot_angle_deg))

    image_file = "{}/jet_image_{}_{}.txt".format(jetpol_run_directory, "i", freq_names[freq_ghz])
    tau_fr_image = np.loadtxt(image_file)
    print(f"Flux = {np.sum(tau_fr_image)}")


    jm_i.load_image_stokes("I", "{}/jet_image_{}_{}.txt".format(jetpol_run_directory, "i", freq_names[freq_ghz]), scale=1.0)
    jm_q.load_image_stokes("Q", "{}/jet_image_{}_{}.txt".format(jetpol_run_directory, "q", freq_names[freq_ghz]), scale=1.0)
    jm_u.load_image_stokes("U", "{}/jet_image_{}_{}.txt".format(jetpol_run_directory, "u", freq_names[freq_ghz]), scale=1.0)

    uvdata.zero_data()
    uvdata.substitute([jm_i, jm_q, jm_u])
    uvdata.noise_add(noise)
    uvdata.save(os.path.join(save_dir, "template_{}.uvf".format(freq_names[freq_ghz])), rewrite=True, downscale_by_freq=False)

#          ╭──────────────────────────────────────────────────────────╮
#          │  Saving model images just convolved with a common beam   │
#          ╰──────────────────────────────────────────────────────────╯
    for stk, jm in zip(("i", "q", "u"), (jm_i, jm_q, jm_u)):
        print("======================================================================================")
        print("      Saving model image (Stokes {}, frequency {} GHz) to difmap format...".format(stk, freq_ghz))
        print("======================================================================================")
        jm.save_image_to_difmap_format(os.path.join(save_dir, "model_dfm_{}_{}.mdl".format(stk.lower(), freq_names[freq_ghz])))
        print("======================================================================================")
        print("     Convolving with common beam ({} mas, {} mas, {} deg) and saving to FITS file...".format(*common_beam))
        print("======================================================================================")
        convert_difmap_model_file_to_CCFITS(os.path.join(save_dir, "model_dfm_{}_{}.mdl".format(stk.lower(), freq_names[freq_ghz])),
                                            stk, common_mapsize_x2, common_beam, template_uvfits,
                                            os.path.join(save_dir, "convolved_{}_{}.fits".format(stk.lower(), freq_names[freq_ghz])))

#          ╭──────────────────────────────────────────────────────────╮
#          │               Plot convolved model images                │
#          ╰──────────────────────────────────────────────────────────╯
        # TODO:
convolved_images_dict = dict()
convolved_pang_arrays = list()
convolved_ipol_arrays = list()
beam_npixels = None

for freq_ghz in freqs_ghz:
    convolved_images_dict[freq_ghz] = dict()
    for stk in stokes:
        convolved_images_dict[freq_ghz][stk] = create_image_from_fits_file(os.path.join(save_dir, "convolved_{}_{}.fits".format(stk, freq_names[freq_ghz])))
        if beam_npixels is None:
            beam_npixels = np.pi*common_beam[0]*common_beam[1]/(4*np.log(2)*common_mapsize_x2[1]**2)

    ipol = convolved_images_dict[freq_ghz]["i"].image
    qpol = convolved_images_dict[freq_ghz]["q"].image
    upol = convolved_images_dict[freq_ghz]["u"].image
    convolved_ipol_arrays.append(ipol)
    convolved_pang_arrays.append(0.5*np.arctan2(upol, qpol))


max_freq = max(freqs_ghz)
ipol = convolved_images_dict[max_freq]["i"].image
qpol = convolved_images_dict[max_freq]["q"].image
upol = convolved_images_dict[max_freq]["u"].image
ppol = np.hypot(qpol, upol)
colors_mask = ppol < ppol.max()*0.001
rm_image, sigma_rotm_array, rotm_chisq_array = rotm_map(np.array(freqs_ghz)*1e9, convolved_pang_arrays, s_chis=None,
                                                          mask=colors_mask, outfile=None, outdir=None,
                                                          mask_on_chisq=False, plot_pxls=None, outfile_pxls=None)
spix_array, sigma_spix_array, spix_chisq_array = spix_map(np.array(freqs_ghz)*1e9, convolved_ipol_arrays, s_flux_maps=None,
                                                          mask=colors_mask, outfile=None, outdir=None,
                                                          mask_on_chisq=False, ampcal_uncertainties=None)


blc, trc = find_bbox(ipol, level=0.001*ipol.max(), min_maxintensity_mjyperbeam=0.1*ipol.max(),min_area_pix=4*beam_npixels, delta=10)

max_rm = np.max(np.abs(rm_image))

fig = iplot(ipol, rm_image, x=convolved_images_dict[max_freq]["i"].x, y=convolved_images_dict[max_freq]["i"].y,
            min_abs_level=0.0001*ipol.max(), colors_mask=colors_mask, color_clim=None, blc=blc, trc=trc,
            beam=common_beam, close=True, colorbar_label=r"RM, rad/m$^2$", show_beam=True, show=False,
            contour_color='black', plot_colorbar=True,
            contour_linewidth=0.25, cmap="bwr")
fig.savefig(os.path.join(save_dir, "RM_Icontours_convolved.png"), dpi=600, bbox_inches="tight")


#          ╭──────────────────────────────────────────────────────────╮
#          │         CLEAN synthetic uv-data (original beams)         │
#          ╰──────────────────────────────────────────────────────────╯
for freq_ghz in freqs_ghz:

    outfname_i = "model_cc_i_{}.fits".format(freq_names[freq_ghz])
    outfname_q = "model_cc_q_{}.fits".format(freq_names[freq_ghz])
    outfname_u = "model_cc_u_{}.fits".format(freq_names[freq_ghz])

    if not do_not_reclean:
        if os.path.exists(os.path.join(save_dir, outfname_i)):
            os.unlink(os.path.join(save_dir, outfname_i))

        # FIXME: Need common uv-range to fix spectral flattening bias!
        clean_difmap(fname="template_{}.uvf".format(freq_names[freq_ghz]), path=save_dir,
                     outfname=outfname_i, outpath=save_dir, stokes="i",
                     mapsize_clean=common_mapsize, path_to_script=path_to_script,
                     show_difmap_output=True)
        clean_difmap(fname="template_{}.uvf".format(freq_names[freq_ghz]), path=save_dir,
                     outfname=outfname_q, outpath=save_dir, stokes="q",
                     mapsize_clean=common_mapsize, path_to_script=path_to_script,
                     show_difmap_output=True)
        clean_difmap(fname="template_{}.uvf".format(freq_names[freq_ghz]), path=save_dir,
                     outfname=outfname_u, outpath=save_dir, stokes="u",
                     mapsize_clean=common_mapsize, path_to_script=path_to_script,
                     show_difmap_output=True)
                     # dfm_model=os.path.join(save_dir, "model_cc_i.mdl"))

#          ╭──────────────────────────────────────────────────────────╮
#          │            Plot single frequency CLEAN image             │
#          ╰──────────────────────────────────────────────────────────╯

    ccimage_i = create_clean_image_from_fits_file(os.path.join(save_dir, outfname_i))
    ccimage_q = create_clean_image_from_fits_file(os.path.join(save_dir, outfname_q))
    ccimage_u = create_clean_image_from_fits_file(os.path.join(save_dir, outfname_u))
    ipol = ccimage_i.image
    qpol = ccimage_q.image
    upol = ccimage_u.image
    ppol = np.hypot(qpol, upol)
    fpol = ppol/ipol
    pang = 0.5*np.arctan2(upol, qpol)
    # Here beam in rad
    beam = ccimage_i.beam
    # Number of pixels in beam
    npixels_beam = np.pi*beam[0]*beam[1]/(4*np.log(2)*mapsizes_dict[freq_ghz][1]**2)

    std = find_image_std(ipol, beam_npixels=npixels_beam)
    print("IPOL image std = {} mJy/beam".format(1000*std))
    blc, trc = find_bbox(ipol, level=4*std, min_maxintensity_mjyperbeam=30*std,
                         min_area_pix=5*npixels_beam, delta=1)
    print("blc = ", blc, ", trc = ", trc)
    if blc[0] == 0: blc = (blc[0]+1, blc[1])
    if blc[1] == 0: blc = (blc[0], blc[1]+1)
    if trc[0] == ipol.shape: trc = (trc[0]-1, trc[1])
    if trc[1] == ipol.shape: trc = (trc[0], trc[1]-1)

    # blc = (300, 300)
    # trc = (800, 700)

    masks_dict, ppol_quantile = pol_mask({"I": ipol, "Q": qpol, "U": upol}, npixels_beam, n_sigma=3,
                                         return_quantile=True)

    # IPOL contours
    # Beam must be in deg
    beam_deg = (beam[0], beam[1], np.rad2deg(beam[2]))
    print("Plotting beam (deg) = ", beam_deg)
    # PPOL contours
    fig = iplot(contours=ppol, x=ccimage_i.x, y=ccimage_i.y, min_abs_level=ppol_quantile,
                blc=blc, trc=trc, beam=beam_deg, close=False,
                contour_color='gray', contour_linewidth=0.25, plot_colorbar=False)
    # Add single IPOL contour and vectors of the PANG
    fig = iplot(contours=ipol, vectors=pang,
                x=ccimage_i.x, y=ccimage_i.y, vinc=4, contour_linewidth=1.0,
                vectors_mask=masks_dict["P"], abs_levels=[2*std], blc=blc, trc=trc,
                beam=beam, close=True, show_beam=True, show=False,
                contour_color='gray', fig=fig, vector_color="black", plot_colorbar=False,
                vector_scale=4)
    fig.savefig(os.path.join(save_dir, "observed_pol_{}.png".format(freq_names[freq_ghz])), dpi=600, bbox_inches="tight")


#          ╭──────────────────────────────────────────────────────────╮
#          │          CLEAN synthetic uv-data (common beam)           │
#          ╰──────────────────────────────────────────────────────────╯
for freq_ghz in freqs_ghz:

    outfname_i = "model_cc_i_{}_common_beam.fits".format(freq_names[freq_ghz])
    outfname_q = "model_cc_q_{}_common_beam.fits".format(freq_names[freq_ghz])
    outfname_u = "model_cc_u_{}_common_beam.fits".format(freq_names[freq_ghz])

    if not do_not_reclean:
        if os.path.exists(os.path.join(save_dir, outfname_i)):
            os.unlink(os.path.join(save_dir, outfname_i))

        # FIXME: Need common uv-range to fix spectral flattening bias!
        clean_difmap(fname="template_{}.uvf".format(freq_names[freq_ghz]), path=save_dir,
                     outfname=outfname_i, outpath=save_dir, stokes="i",
                     mapsize_clean=common_mapsize, path_to_script=path_to_script,
                     show_difmap_output=True, beam_restore=common_beam,
                     uvrange=common_uvrange)
        clean_difmap(fname="template_{}.uvf".format(freq_names[freq_ghz]), path=save_dir,
                     outfname=outfname_q, outpath=save_dir, stokes="q",
                     mapsize_clean=common_mapsize, path_to_script=path_to_script,
                     show_difmap_output=True, beam_restore=common_beam,
                     uvrange=common_uvrange)
        clean_difmap(fname="template_{}.uvf".format(freq_names[freq_ghz]), path=save_dir,
                     outfname=outfname_u, outpath=save_dir, stokes="u",
                     mapsize_clean=common_mapsize, path_to_script=path_to_script,
                     show_difmap_output=True, beam_restore=common_beam,
                     uvrange=common_uvrange)


ccimages = dict()
pang_arrays = list()
sigma_pang_arrays = list()
ipol_arrays = list()
sigma_ipol_arrays = list()
beam_npixels = None


for freq_ghz in freqs_ghz:
    ccimages[freq_ghz] = dict()
    for stk in stokes:
        ccimages[freq_ghz][stk] = create_clean_image_from_fits_file(os.path.join(save_dir, "model_cc_{}_{}_common_beam.fits".format(stk, freq_names[freq_ghz])))
        if beam_npixels is None:
            beam = ccimages[freq_ghz][stk].beam
            beam_npixels = np.pi*beam[0]*beam[1]/(4*np.log(2)*common_mapsize[1]**2)

    ipol = ccimages[freq_ghz]["i"].image
    qpol = ccimages[freq_ghz]["q"].image
    upol = ccimages[freq_ghz]["u"].image
    ipol_arrays.append(ipol)
    pang_arrays.append(0.5*np.arctan2(upol, qpol))
    sigma_pang_array, sigma_ppol_array = hovatta_find_sigma_pang(ccimages[freq_ghz]["q"],
                                                                 ccimages[freq_ghz]["u"],
                                                                 ccimages[freq_ghz]["i"],
                                                                 sigma_evpa=0, d_term=0, n_ant=1, n_if=1, n_scan=1)
    sigma_pang_arrays.append(sigma_pang_array)
    mask_dict, ppol_quantile = pol_mask({stk.upper(): ccimages[freq_ghz][stk].image for stk in stokes}, beam_npixels, n_sigma=2,
                                        return_quantile=True)
    ccimages[freq_ghz]["masks"] = mask_dict
    ccimages[freq_ghz]["pquantile"] = ppol_quantile

    std = find_image_std(ipol, beam_npixels=beam_npixels)
    sigma_ipol_arrays.append(np.ones(ipol.shape)*std)
    blc, trc = find_bbox(ipol, level=4*std, min_maxintensity_mjyperbeam=10*std,min_area_pix=4*beam_npixels, delta=1)
    if blc[0] == 0: blc = (blc[0]+1, blc[1])
    if blc[1] == 0: blc = (blc[0], blc[1]+1)
    if trc[0] == ipol.shape: trc = (trc[0]-1, trc[1])
    if trc[1] == ipol.shape: trc = (trc[0], trc[1]-1)

    ccimages[freq_ghz]["box"] = (blc, trc)
    ccimages[freq_ghz]["std"] = std


common_pmask = np.logical_or.reduce([ccimages[freq]["masks"]["P"] for freq in freqs_ghz])
common_imask = np.logical_or.reduce([ccimages[freq]["masks"]["I"] for freq in freqs_ghz])
# FIXME: Check if the fit is linear!!!
rotm_array, sigma_rotm_array, rotm_chisq_array = rotm_map(np.array(freqs_ghz)*1e9, pang_arrays, sigma_pang_arrays,
                                                          mask=common_pmask, outfile=None, outdir=None,
                                                          mask_on_chisq=True, plot_pxls=None, outfile_pxls=None)
spix_array, sigma_spix_array, spix_chisq_array = spix_map(np.array(freqs_ghz)*1e9, ipol_arrays, sigma_ipol_arrays,
                                                          mask=common_imask, outfile=None, outdir=None,
                                                          mask_on_chisq=False, ampcal_uncertainties=None)

plot_freq = min(freqs_ghz)
ipol = ccimages[plot_freq]["i"].image
blc, trc = ccimages[plot_freq]["box"]
fig = iplot(ipol, rotm_array, x=ccimages[plot_freq]["i"].x, y=ccimages[plot_freq]["i"].y,
            min_abs_level=3*ccimages[plot_freq]["std"], colors_mask=common_pmask, color_clim=None, blc=blc, trc=trc,
            beam=common_beam, close=True, colorbar_label=r"RM, rad/m$^2$", show_beam=True, show=False,
            contour_color='black', plot_colorbar=True,
            contour_linewidth=0.25, cmap="bwr")
fig.savefig(os.path.join(save_dir, "rotm_vlbi.png"), dpi=600, bbox_inches="tight")

fig = iplot(ipol, spix_array, x=ccimages[plot_freq]["i"].x, y=ccimages[plot_freq]["i"].y,
            min_abs_level=3*ccimages[min(freqs_ghz)]["std"], colors_mask=common_imask, color_clim=[-1., 0], blc=blc, trc=trc,
            beam=common_beam, close=True, colorbar_label=r"$\alpha$", show_beam=True, show=False,
            cmap='bwr', contour_color='black', plot_colorbar=True,
            contour_linewidth=0.25)
fig.savefig(os.path.join(save_dir, "spix_vlbi.png"), dpi=600, bbox_inches="tight")
