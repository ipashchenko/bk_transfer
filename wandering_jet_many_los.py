import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import sys
from jet_image import JetImage, TwinJetImage
from vlbi_utils import find_image_std, find_bbox, pol_mask, correct_ppol_bias
sys.path.insert(0, '../ve/vlbi_errors')
from uv_data import UVData
from spydiff import clean_difmap, find_nw_beam
from from_fits import create_clean_image_from_fits_file
from image import plot as iplot
sys.path.insert(0, '../stackemall')
from stack_utils import stat_of_masked


freq_ghz = 15.4
# Directory to save files
save_dir = "/home/ilya/github/bk_transfer/results/new"
# Some template UVFITS with full polarization. Its uv-coverage and noise will be used while creating fake data
template_uvfits = "/home/ilya/github/bk_transfer/uvfits/1458+718.u.2006_09_06.uvf"
# Multiplicative factor for noise added to model visibilities.
noise_scale_factor = 1.0
# Used in CLEAN
mapsize = (512, 0.1)
n_epochs = 20


nw_beam = find_nw_beam(template_uvfits, "i", mapsize=mapsize)
nw_beam_size = np.sqrt(nw_beam[0]*nw_beam[1])

# Common beam
common_beam = (nw_beam_size, nw_beam_size, 0)
print("NW beam = ", nw_beam_size)
exec_dir = "/home/ilya/github/bk_transfer/Release"

# C++ code run parameters
# z = 0.00436
z = 0.1
# n_along = 1000
# n_across = 100
# lg_pixel_size_min_mas = -4
# lg_pixel_size_max_mas = -1

n_along = 300
n_across = 100
lg_pixel_size_min_mas = -2
lg_pixel_size_max_mas = -0.5

resolutions = np.logspace(lg_pixel_size_min_mas, lg_pixel_size_max_mas, n_along)
print("Model jet extends up to {:.1f} mas!".format(np.sum(resolutions)))
##############################################
# No need to change anything below this line #
##############################################

# Plot only jet emission and do not plot counter-jet?
jet_only = True
path_to_script = "/home/ilya/github/bk_transfer/scripts/script_clean_nw"
parallels_run_file = os.path.join(save_dir, "parallels_run.txt")

images_i = list()
images_q = list()
images_u = list()
images_pang = list()

# For 0.1/Gamma
# B_1 = 0.7
B_1 = 0.4
# For 0.3/Gamma
# B_1 = 0.4
m_b = 1.0
# Does not matter for EQ
K_1 = 1000.
Gamma = 10.
LOS_coeff = 0.1
HOAngle_deg = 15.
los_angle_deg = np.round(np.rad2deg(np.arcsin(LOS_coeff/Gamma)), 2)
cone_half_angle_deg = np.round(np.rad2deg(np.arctan(np.tan(np.deg2rad(HOAngle_deg)) * np.sin(np.deg2rad(los_angle_deg)))), 2)
print(f"LOS(deg) = {los_angle_deg}")
print(f"Cone HA (deg) = {cone_half_angle_deg}")

# sys.exit(0)

rot_angles_deg = list()
with open(f"{parallels_run_file}", "w") as fo:
    for t, i in enumerate(range(n_epochs)):
        # -107 for M87
        rot_angle_deg = np.random.uniform(-15, 15, 1)[0]
        rot_angles_deg.append(rot_angle_deg)

        fo.write("{} {} {} {} {} {} {} {} {} {} {} {:.1f} 0 0 0 1\n".format(z, los_angle_deg, cone_half_angle_deg,
                                                                            B_1, m_b, K_1, Gamma, n_along, n_across,
                                                                            lg_pixel_size_min_mas, lg_pixel_size_max_mas,
                                                                            t))

        # args = (z, los_angle_deg, cone_half_angle_deg, B_1, m_b, K_1, Gamma, n_along, n_across, lg_pixel_size_min_mas, lg_pixel_size_max_mas, 0, 0, 0, 0, 1)

np.savetxt(os.path.join(save_dir, "rot_angles.txt"), rot_angles_deg)
sys.exit(0)
os.chdir(exec_dir)
n_jobs = 2
os.system("parallel --files --results n_{12}" + f" --joblog log --jobs {n_jobs} -a {parallels_run_file} -n 1 -m --colsep ' ' \"./bk_transfer\"")

# print(*args)
# os.system("./bk_transfer {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format(*args))
# sys.exit(0)



for t, i in enumerate(range(n_epochs)):
    # Substitute real data
    uvdata = UVData(template_uvfits)
    noise = uvdata.noise(average_freq=False, use_V=False)
    # If one needs to decrease the noise this is the way to do it
    for baseline, baseline_noise_std in noise.items():
        noise.update({baseline: noise_scale_factor*baseline_noise_std})

    print(noise)

    stokes = ("I", "Q", "U", "V")
    jms = [JetImage(z=z, n_along=n_along, n_across=n_across,
                    lg_pixel_size_mas_min=lg_pixel_size_min_mas, lg_pixel_size_mas_max=lg_pixel_size_max_mas,
                    jet_side=True, rot=np.deg2rad(rot_angle_deg)) for _ in stokes]
    # cjms = [JetImage(z=z, n_along=n_along, n_across=n_across,
    #                  lg_pixel_size_mas_min=lg_pixel_size_mas_min, lg_pixel_size_mas_max=lg_pixel_size_mas_max,
    #                  jet_side=False) for _ in stokes]
    for i, stk in enumerate(stokes):
        jms[i].load_image_stokes(stk, "{}/jet_image_{}_u_{:.1}.txt".format(exec_dir, stk.lower(), t), scale=1.0)
        # cjms[i].load_image_stokes(stk, "../{}/cjet_image_{}_{}.txt".format(jetpol_run_directory, stk.lower(), freq_ghz), scale=1.0)

    # List of models (for J & CJ) for all stokes
    # js = [TwinJetImage(jms[i], cjms[i]) for i in range(len(stokes))]

    uvdata.zero_data()
    if jet_only:
        uvdata.substitute(jms)
    else:
        # uvdata.substitute(js)
        pass
    # Optionally
    uvdata.rotate_evpa(np.deg2rad(rot_angle_deg))
    uvdata.noise_add(noise)
    uvdata.save(os.path.join(save_dir, "template.uvf"), rewrite=True)

    # CLEAN synthetic UV-data
    for stk in stokes:
        outfname = "model_cc_{}.fits".format(stk.lower())
        if os.path.exists(os.path.join(save_dir, outfname)):
            os.unlink(os.path.join(save_dir, outfname))
        clean_difmap(fname="template.uvf", path=save_dir,
                     outfname=outfname, outpath=save_dir, stokes=stk.lower(),
                     mapsize_clean=mapsize, path_to_script=path_to_script,
                     show_difmap_output=True,
                     # text_box=text_boxes[freq],
                     dfm_model=os.path.join(save_dir, "model_cc_{}.mdl".format(stk.lower())),
                     beam_restore=common_beam)

    ccimages = {stk: create_clean_image_from_fits_file(os.path.join(save_dir, "model_cc_{}.fits".format(stk.lower())))
                for stk in stokes}
    ipol = ccimages["I"].image
    beam = ccimages["I"].beam
    # Number of pixels in beam
    npixels_beam = np.pi*beam[0]*beam[1]/(4*np.log(2)*mapsize[1]**2)

    std = find_image_std(ipol, beam_npixels=npixels_beam)
    print("IPOL image std = {} mJy/beam".format(1000*std))
    blc, trc = find_bbox(ipol, level=4*std, min_maxintensity_mjyperbeam=10*std,
                         min_area_pix=4*npixels_beam, delta=10)
    if blc[0] == 0: blc = (blc[0]+1, blc[1])
    if blc[1] == 0: blc = (blc[0], blc[1]+1)
    if trc[0] == ipol.shape: trc = (trc[0]-1, trc[1])
    if trc[1] == ipol.shape: trc = (trc[0], trc[1]-1)
    masks_dict, ppol_quantile = pol_mask({stk: ccimages[stk].image for stk in
                                          ("I", "Q", "U")}, npixels_beam, n_sigma=4,
                                         return_quantile=True)
    ppol = np.hypot(ccimages["Q"].image, ccimages["U"].image)
    ppol = correct_ppol_bias(ipol, ppol, ccimages["Q"].image, ccimages["U"].image, npixels_beam)
    pang = 0.5*np.arctan2(ccimages["U"].image, ccimages["Q"].image)
    fpol = ppol/ipol

    images_i.append(ipol)
    images_q.append(ccimages["Q"].image)
    images_u.append(ccimages["U"].image)
    images_pang.append(np.ma.array(pang, mask=masks_dict["P"]))


stack_i = np.mean(images_i, axis=0)
stack_q = np.mean(images_q, axis=0)
stack_u = np.mean(images_u, axis=0)
stack_p = np.hypot(stack_q, stack_u)
stack_pang = 0.5*np.arctan2(stack_u, stack_q)
stack_pang_std = stat_of_masked(images_pang, stat="scipy_circstd", n_epochs_not_masked_min=3)
stack_p = correct_ppol_bias(stack_i, stack_p, stack_q, stack_u, npixels_beam)
stack_fpol = stack_p/stack_i
masks_dict, ppol_quantile = pol_mask({"I": stack_i, "Q": stack_q, "U": stack_u}, npixels_beam, n_sigma=4,
                                     return_quantile=True)
std = find_image_std(stack_i, beam_npixels=npixels_beam)
blc, trc = find_bbox(stack_i, level=4*std, min_maxintensity_mjyperbeam=10*std,
                     min_area_pix=4*npixels_beam, delta=10)

# IPOL contours
fig = iplot(stack_i, x=ccimages["I"].x, y=ccimages["I"].y,
            min_abs_level=4*std, blc=blc, trc=trc, beam=beam, close=True, show_beam=True, show=False,
            contour_color='gray', contour_linewidth=0.25)
fig.savefig(os.path.join(save_dir, "observed_stack_i.png"), dpi=600, bbox_inches="tight")

# PPOL contours
fig = iplot(stack_p, x=ccimages["I"].x, y=ccimages["I"].y,
            min_abs_level=ppol_quantile, blc=blc, trc=trc,
            close=False, contour_color='black',
            plot_colorbar=False)
# Add single IPOL contour and vectors of the PANG
fig = iplot(contours=stack_i, vectors=stack_pang,
            x=ccimages["I"].x, y=ccimages["I"].y, vinc=4, contour_linewidth=0.25,
            vectors_mask=masks_dict["P"], abs_levels=[3*std], blc=blc, trc=trc,
            beam=beam, close=True, show_beam=True, show=False,
            contour_color='gray', fig=fig, vector_color="black", plot_colorbar=False)
axes = fig.get_axes()[0]
axes.invert_xaxis()
fig.savefig(os.path.join(save_dir, "observed_stack_p.png"), dpi=600, bbox_inches="tight")

fig = iplot(stack_i, stack_fpol, x=ccimages["I"].x, y=ccimages["I"].y,
            min_abs_level=4*std, colors_mask=masks_dict["P"], color_clim=[0, 0.7], blc=blc, trc=trc,
            beam=beam, close=True, colorbar_label="m", show_beam=True, show=False,
            cmap='gnuplot', contour_color='black', plot_colorbar=True,
            contour_linewidth=0.25)
fig.savefig(os.path.join(save_dir, "observed_stack_fpol.png"), dpi=600, bbox_inches="tight")

fig = iplot(stack_i, np.rad2deg(stack_pang_std), x=ccimages["I"].x, y=ccimages["I"].y,
            min_abs_level=4*std, colors_mask=masks_dict["P"], color_clim=None, blc=blc, trc=trc,
            beam=beam, close=True, colorbar_label=r"$\sigma_{\rm EVPA},$ $^{\circ}$", show_beam=True, show=False,
            cmap='gnuplot', contour_color='black', plot_colorbar=True,
            contour_linewidth=0.25)
fig.savefig(os.path.join(save_dir, "observed_stack_pang_std.png"), dpi=600, bbox_inches="tight")