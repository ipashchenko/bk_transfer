import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/ilya/github/ve/vlbi_errors')
from uv_data import get_uvrange
from spydiff import find_nw_beam, clean_difmap, CLEAN_difmap, find_bbox
from utils import find_image_std
from from_fits import create_clean_image_from_fits_file, create_image_from_fits_file
from image import plot as iplot
from image_ops import spix_map
import warnings
from astropy import wcs
warnings.filterwarnings("ignore", category=wcs.FITSFixedWarning)


from matplotlib.colors import LinearSegmentedColormap
colors = [# almost black
    (0.0, 0.0, 0.2),
    # dark blue
    (0.0, 0.0, 0.4),
    # blue
    (0.0, 0.0, 0.8),
    (0.19, 0.88, 0.8),
    (0.44, 0.60, 0.74),
    # green
    (0, 0.7, 0),
    # yellow (-0.25 for (-1.5, 1) color range)
    (1.00, 1.00, 0.01),
    (0.9, 0, 0),
    # red
    (1, 0, 0),
    (0.80, 0.13, 0.00),
    (0.57, 0.35, 0.35),
    (0.55, 0.00, 1.00),
    # white
    (1 , 1, 1)]
# colors = [# almost black
#           (0.0, 0.0, 0.2),
#           # dark blue
#           (0.0, 0.0, 0.4),
#           # blue
#           (0.0, 0.0, 0.8),
#           (0.19, 0.88, 0.8),
#           (0.44, 0.60, 0.74),
#           # green
#           (0, 0.7, 0),
#           # yellow (-0.25)
#           (1.00, 1.00, 0.01),
#           (0.9, 0, 0),
#           # red
#           (1, 0, 0),
#           (0.80, 0.13, 0.00),
#           (0.57, 0.35, 0.35),
#           (0.55, 0.00, 1.00),
#           # white
#           (1 , 1, 1)]
n_bin = 200  # Discretizes the interpolation into bins
cmap_name = 'my_list'
# Create the colormap
cm_my = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)

data_dir = "/home/ilya/data/alpha/MOJAVE"
save_dir = data_dir
source = "1807+698"
do_clean = True
deep_factor = 1.0
wins = None
# wins = os.path.join(data_dir, f"wins_{source}.txt")

if source == "1228+126":
    uvfits_u = "/home/ilya/data/alpha/MOJAVE/1228+126.u.2006_06_15.uvf"
    uvfits_x = "/home/ilya/data/alpha/MOJAVE/1228+126.x.2006_06_15.uvf"
    uvfits_y = "/home/ilya/data/alpha/MOJAVE/1228+126.y.2006_06_15.uvf"
    uvfits_j = "/home/ilya/data/alpha/MOJAVE/1228+126.j.2006_06_15.uvf"
elif source == "0055+300":
    uvfits_u = "/home/ilya/data/alpha/MOJAVE/0055+300.u.2006_02_12.uvf"
    uvfits_x = "/home/ilya/data/alpha/MOJAVE/0055+300.x.2006_02_12.uvf"
    uvfits_y = "/home/ilya/data/alpha/MOJAVE/0055+300.y.2006_02_12.uvf"
    uvfits_j = "/home/ilya/data/alpha/MOJAVE/0055+300.j.2006_02_12.uvf"
elif source == "1652+398":
    uvfits_u = "/home/ilya/data/alpha/MOJAVE/1652+398.u.2006_02_12.uvf"
    uvfits_x = "/home/ilya/data/alpha/MOJAVE/1652+398.x.2006_02_12.uvf"
    uvfits_y = "/home/ilya/data/alpha/MOJAVE/1652+398.y.2006_02_12.uvf"
    uvfits_j = "/home/ilya/data/alpha/MOJAVE/1652+398.j.2006_02_12.uvf"
elif source == "1633+382":
    uvfits_u = "/home/ilya/data/alpha/MOJAVE/1633+382.u.2006_09_06.uvf"
    uvfits_x = "/home/ilya/data/alpha/MOJAVE/1633+382.x.2006_09_06.uvf"
    uvfits_y = "/home/ilya/data/alpha/MOJAVE/1633+382.y.2006_09_06.uvf"
    uvfits_j = "/home/ilya/data/alpha/MOJAVE/1633+382.j.2006_09_06.uvf"
elif source == "1641+399":
    uvfits_u = "/home/ilya/data/alpha/MOJAVE/1641+399.u.2006_06_15.uvf"
    uvfits_x = "/home/ilya/data/alpha/MOJAVE/1641+399.x.2006_06_15.uvf"
    uvfits_y = "/home/ilya/data/alpha/MOJAVE/1641+399.y.2006_06_15.uvf"
    uvfits_j = "/home/ilya/data/alpha/MOJAVE/1641+399.j.2006_06_15.uvf"
elif source == "1749+701":
    uvfits_u = "/home/ilya/data/alpha/MOJAVE/1749+701.u.2006_04_05.uvf"
    uvfits_x = "/home/ilya/data/alpha/MOJAVE/1749+701.x.2006_04_05.uvf"
    uvfits_y = "/home/ilya/data/alpha/MOJAVE/1749+701.y.2006_04_05.uvf"
    uvfits_j = "/home/ilya/data/alpha/MOJAVE/1749+701.j.2006_04_05.uvf"
elif source == "1807+698":
    uvfits_u = "/home/ilya/data/alpha/MOJAVE/1807+698.u.2006_02_12.uvf"
    uvfits_x = "/home/ilya/data/alpha/MOJAVE/1807+698.x.2006_02_12.uvf"
    uvfits_y = "/home/ilya/data/alpha/MOJAVE/1807+698.y.2006_02_12.uvf"
    uvfits_j = "/home/ilya/data/alpha/MOJAVE/1807+698.j.2006_02_12.uvf"
elif source == "2021+317":
    uvfits_u = "/home/ilya/data/alpha/MOJAVE/2021+317.u.2006_08_09.uvf"
    uvfits_x = "/home/ilya/data/alpha/MOJAVE/2021+317.x.2006_08_09.uvf"
    uvfits_y = "/home/ilya/data/alpha/MOJAVE/2021+317.y.2006_08_09.uvf"
    uvfits_j = "/home/ilya/data/alpha/MOJAVE/2021+317.j.2006_08_09.uvf"
else:
    raise Exception

common_mapsize = (1024, 0.1)


if source == "1228+126":
    # Shift X-U.
    shift_value = 0.18
    deg = 15
    shift_x = np.array([-shift_value * np.cos(np.deg2rad(deg)), shift_value * np.sin(np.deg2rad(deg))])
    shift_y = shift_x
    shift_j = 0.3*shift_x
elif source == "0055+300":
    # Source --- Epoch "YYYY-MM-DD" PAM deg PA1 deg Dr1 mas Drp1 mas PA2 deg Dr2 mas Drp2 mas PA3 deg Dr3 mas Drp3 mas
    # 0055+300 2006-02-12 -50.1 -45.2 0.179 0.179 -10.2 0.083 0.064 -61.5 0.053 0.052
    # Shift X-U.
    shift_value = 0.19
    deg = 60
    shift_x = np.array([-shift_value * np.cos(np.deg2rad(deg)), shift_value * np.sin(np.deg2rad(deg))])
    shift_y = shift_x
    shift_j = 0.3*shift_x
elif source == "1652+398":
    # Source --- Epoch "YYYY-MM-DD" PAM deg PA1 deg Dr1 mas Drp1 mas PA2 deg Dr2 mas Drp2 mas PA3 deg Dr3 mas Drp3 mas
    # 1652+398	2006-02-12	147.9	171.5	0.289	0.265	171.5	0.269	0.246	160.4	0.200	0.196
    # Shift X-U.
    shift_value = 0.27
    deg = 171.5 - 90
    shift_x = np.array([shift_value * np.cos(np.deg2rad(deg)), -shift_value * np.sin(np.deg2rad(deg))])
    shift_y = shift_x
    shift_j = 0.3*shift_x
elif source == "1633+382":
    # Source --- Epoch "YYYY-MM-DD" PAM deg PA1 deg Dr1 mas Drp1 mas PA2 deg Dr2 mas Drp2 mas PA3 deg Dr3 mas Drp3 mas
    # 1633+382 2006-09-06 -72.8 -66.6 0.119 0.119 -68.3 0.158 0.157 -66.9 0.157 0.156
    # Shift X-U.
    shift_value = 0.12
    deg = 90-66.6
    shift_x = np.array([-shift_value * np.cos(np.deg2rad(deg)), shift_value * np.sin(np.deg2rad(deg))])
    shift_y = shift_x
    shift_j = 0.3*shift_x
elif source == "1641+399":
    # Source --- Epoch "YYYY-MM-DD" PAM deg PA1 deg Dr1 mas Drp1 mas PA2 deg Dr2 mas Drp2 mas PA3 deg Dr3 mas Drp3 mas
    # 1641+399 2006-06-15 -90.5 -90.8 0.211 0.211 -91.5 0.190 0.190 -89.0 0.121 0.121
    # Shift X-U.
    shift_value = 0.211
    deg = 0.0
    shift_x = np.array([-shift_value * np.cos(np.deg2rad(deg)), shift_value * np.sin(np.deg2rad(deg))])
    shift_y = shift_x
    shift_j = 0.3*shift_x
elif source == "1749+701":
    # Source --- Epoch "YYYY-MM-DD" PAM deg PA1 deg Dr1 mas Drp1 mas PA2 deg Dr2 mas Drp2 mas PA3 deg Dr3 mas Drp3 mas
    # 1749+701 2006-04-05 -59.4 -34.9 0.196 0.179 -64.0 0.224 0.223 -37.5 0.184 0.170
    # Shift X-U.
    shift_value = 0.185
    deg = 90 - 55.
    shift_x = np.array([-shift_value * np.cos(np.deg2rad(deg)), shift_value * np.sin(np.deg2rad(deg))])
    shift_y = shift_x
    shift_j = 0.3*shift_x
elif source == "1807+698":
    # Source --- Epoch "YYYY-MM-DD" PAM deg PA1 deg Dr1 mas Drp1 mas PA2 deg Dr2 mas Drp2 mas PA3 deg Dr3 mas Drp3 mas
    # 1807+698 2006-02-12 -101.3 -104.9 0.249 0.248 -92.8 0.186 0.184 73.3 0.019 -0.019
    # Shift X-U.
    shift_value = 0.25
    deg = -15
    shift_x = np.array([-shift_value * np.cos(np.deg2rad(deg)), shift_value * np.sin(np.deg2rad(deg))])
    shift_y = shift_x
    shift_j = 0.3*shift_x
elif source == "2021+317":
    # Source --- Epoch "YYYY-MM-DD" PAM deg PA1 deg Dr1 mas Drp1 mas PA2 deg Dr2 mas Drp2 mas PA3 deg Dr3 mas Drp3 mas
    # 2021+317 2006-08-09 156.5 173.5 0.384 0.367 157.9 0.218 0.218 166.3 0.120 0.118
    # Shift X-U.
    shift_value = 0.37
    deg = -90
    shift_x = np.array([-shift_value * np.cos(np.deg2rad(deg)), shift_value * np.sin(np.deg2rad(deg))])
    shift_y = shift_x
    shift_j = 0.3*shift_x
else:
    raise Exception("Provide image shifts for this source!")


# uvrange_x = get_uvrange(uvfits_x)
# uvrange_u = get_uvrange(uvfits_u)
# common_uvrange = (uvrange_u[0], uvrange_x[1])
if source == "0055+300":
    common_uvrange = (4.547, 232.1)
elif source == "1228+126":
    common_uvrange = (8.915, 232.4)
elif source == "1652+398":
    common_uvrange = (4.561, 233.05)
elif source == "1633+382":
    common_uvrange = (5.3, 233.03)
elif source == "1641+399":
    common_uvrange = (4.1, 231.7)
elif source == "1749+701":
    common_uvrange = (7.112, 231.5)
elif source == "1807+698":
    common_uvrange = (7.15, 233.1)
elif source == "2021+317":
    common_uvrange = (14.63, 232.5)
else:
    raise Exception("Provide uvrange for this source!")

nw_beam_x = find_nw_beam(uvfits_x, stokes="I", mapsize=(1024, 0.1), uv_range=None, working_dir=None)

if wins is not None:
    base_name = f"{source}_deep_{deep_factor}_wins"
else:
    base_name = f"{source}_deep_{deep_factor}_nowins"
if do_clean:
    # CLEAN with common mapsize and convolving beam
    if not os.path.exists(os.path.join(data_dir, f"{base_name}_x.fits")):
    # if True:
        CLEAN_difmap(uvfits_x, "i", common_mapsize, f"{base_name}_x.fits", restore_beam=nw_beam_x,
                     boxfile=wins, working_dir=data_dir, uvrange=common_uvrange,
                     box_clean_nw_niter=500, clean_gain=0.03, dynam_su=20, dynam_u=8, deep_factor=deep_factor,
                     remove_difmap_logs=True, save_noresid=None, save_resid_only=None, save_dfm=None,
                     noise_to_use="V", shift=shift_x)
        CLEAN_difmap(uvfits_u, "i", common_mapsize, f"{base_name}_u.fits", restore_beam=nw_beam_x,
                     boxfile=wins, working_dir=data_dir, uvrange=common_uvrange,
                     box_clean_nw_niter=500, clean_gain=0.03, dynam_su=20, dynam_u=8, deep_factor=deep_factor,
                     remove_difmap_logs=True, save_noresid=None, save_resid_only=None, save_dfm=None,
                     noise_to_use="V")
        CLEAN_difmap(uvfits_y, "i", common_mapsize, f"{base_name}_y.fits", restore_beam=nw_beam_x,
                     boxfile=wins, working_dir=data_dir, uvrange=common_uvrange,
                     box_clean_nw_niter=500, clean_gain=0.03, dynam_su=20, dynam_u=8, deep_factor=deep_factor,
                     remove_difmap_logs=True, save_noresid=None, save_resid_only=None, save_dfm=None,
                     noise_to_use="V", shift=shift_y)
        CLEAN_difmap(uvfits_j, "i", common_mapsize, f"{base_name}_j.fits", restore_beam=nw_beam_x,
                     boxfile=wins, working_dir=data_dir, uvrange=common_uvrange,
                     box_clean_nw_niter=500, clean_gain=0.03, dynam_su=20, dynam_u=8, deep_factor=deep_factor,
                     remove_difmap_logs=True, save_noresid=None, save_resid_only=None, save_dfm=None,
                     noise_to_use="V", shift=shift_j)

    # clean_difmap(fname=uvfits_x, path=save_dir, outfname="mojave_common_x.fits", outpath=save_dir, stokes="i",
    #              mapsize_clean=common_mapsize, path_to_script=path_to_script, show_difmap_output=True,
    #              beam_restore=nw_beam_x, shift=shift_x)
    # clean_difmap(fname=uvfits_u, path=save_dir, outfname="mojave_common_u.fits", outpath=save_dir, stokes="i",
    #              mapsize_clean=common_mapsize, path_to_script=path_to_script, show_difmap_output=True,
    #              beam_restore=nw_beam_x)
    # clean_difmap(fname=uvfits_y, path=save_dir, outfname="mojave_common_y.fits", outpath=save_dir, stokes="i",
    #              mapsize_clean=common_mapsize, path_to_script=path_to_script, show_difmap_output=True,
    #              beam_restore=nw_beam_x, shift=shift_y)
    # clean_difmap(fname=uvfits_j, path=save_dir, outfname="mojave_common_j.fits", outpath=save_dir, stokes="i",
    #              mapsize_clean=common_mapsize, path_to_script=path_to_script, show_difmap_output=True,
    #              beam_restore=nw_beam_x, shift=shift_j)

ccimage_x = create_image_from_fits_file(os.path.join(data_dir, f"{base_name}_x.fits"))
ccimage_u = create_image_from_fits_file(os.path.join(data_dir, f"{base_name}_u.fits"))
ccimage_y = create_image_from_fits_file(os.path.join(data_dir, f"{base_name}_y.fits"))
ccimage_j = create_image_from_fits_file(os.path.join(data_dir, f"{base_name}_j.fits"))
ipol_low = ccimage_x.image
ipol_high = ccimage_u.image
# Number of pixels in beam
npixels_beam = np.pi*nw_beam_x[0]*nw_beam_x[1]/(4*np.log(2)*common_mapsize[1]**2)
std_low = find_image_std(ipol_low, beam_npixels=npixels_beam)
std_y = find_image_std(ccimage_y.image, beam_npixels=npixels_beam)
std_j = find_image_std(ccimage_j.image, beam_npixels=npixels_beam)
std_high = find_image_std(ipol_high, beam_npixels=npixels_beam)

std = std_low
std_low = np.sqrt(std_low**2 + (1.5*std_low)**2)
std_high = np.sqrt(std_high**2 + (1.5*std_high)**2)
std_y = np.sqrt(std_y**2 + (1.5*std_y)**2)
std_j = np.sqrt(std_j**2 + (1.5*std_j)**2)

mask_low = ipol_low < 3*std_low
mask_high = ipol_high < 3*std_high
mask_j = ccimage_j.image < 3*std_j
mask_y = ccimage_y.image < 3*std_y


# blc, trc = find_bbox(ipol_low, level=3*std_low, min_maxintensity_mjyperbeam=10*std_low,
#                      min_area_pix=10*npixels_beam, delta=10)
# print(blc, trc)
if source == "0055+300":
    blc = (470, 470)
    trc = (620, 620)
elif source == "1228+126":
    blc = (470, 470)
    trc = (870, 650)
elif source == "1652+398":
    blc = (264, 347)
    trc = (568, 569)
elif source == "1633+382":
    blc = (469, 464)
    trc = (595, 575)
elif source == "1641+399":
    blc = (459, 455)
    trc = (730, 676)
elif source == "1749+701":
    blc = (473, 478)
    trc = (585, 583)
elif source == "1807+698":
    blc = (480, 467)
    trc = (680, 540)
elif source == "2021+317":
    blc = (485, 437)
    trc = (546, 547)
else:
    raise Exception("Find blc,trc for this source!")

common_imask_all = np.logical_or.reduce([mask_low, mask_y, mask_j, mask_high])
np.savetxt(os.path.join(data_dir, f"mask_{base_name}.txt"), common_imask_all)
# common_imask_all = np.loadtxt(os.path.join(data_dir, "mask_4freqs.txt"))
# spix_array = np.log(ipol_low/ipol_high)/np.log(8.1/15.4)

one = np.ones(ipol_low.shape)
spix_array, sigma_spix_array, spix_chisq_array = spix_map(np.array([8.1, 8.4, 12.1, 15.4])*1e9,
                                                          [ccimage_x.image, ccimage_y.image, ccimage_j.image, ccimage_u.image],
                                                          [std_low*one, std_y*one, std_j*one, std_high*one],
                                                          mask=common_imask_all, outfile=None, outdir=None,
                                                          mask_on_chisq=False, ampcal_uncertainties=None)
np.savetxt(os.path.join(data_dir, f"spix_{base_name}.txt"), spix_array)
np.savetxt(os.path.join(data_dir, f"spix_error_{base_name}.txt"), sigma_spix_array)

fig = iplot(ipol_low, spix_array, x=ccimage_x.x, y=ccimage_x.y,
            min_abs_level=3*std, colors_mask=common_imask_all, color_clim=[-2.0, 1.0], blc=blc, trc=trc,
            beam=nw_beam_x, close=True, colorbar_label=r"$\alpha$", show_beam=True, show=False,
            cmap=cm_my, contour_color='black', plot_colorbar=True,
            contour_linewidth=0.25, beam_place="lr")
if wins is not None:
    fig.savefig(os.path.join(data_dir, f"{source}_spix_XYJU_I_X_deep_{deep_factor}_wins.png"), dpi=600, bbox_inches="tight")
else:
    fig.savefig(os.path.join(data_dir, f"{source}_spix_XYJU_I_X_deep_{deep_factor}_nowins.png"), dpi=600, bbox_inches="tight")



# Plot SPIX difference map - run with no wins and deep_factor = 1
base_name_deep = f"{source}_deep_0.1_wins"
spix_array_deep = np.loadtxt(os.path.join(data_dir, f"spix_{base_name_deep}.txt"))
sigma_spix_array_deep = np.loadtxt(os.path.join(data_dir, f"spix_error_{base_name_deep}.txt"))
common_imask_all_deep = np.loadtxt(os.path.join(data_dir, f"mask_{base_name_deep}.txt"))

fig = iplot(ipol_low, spix_array - spix_array_deep, x=ccimage_x.x, y=ccimage_x.y,
            min_abs_level=3*std, colors_mask=common_imask_all_deep, color_clim=[-0.5, 0.5], blc=blc, trc=trc,
            beam=nw_beam_x, close=True, colorbar_label=r"$\alpha - \alpha_{\rm deep}$", show_beam=True, show=False,
            cmap="bwr", contour_color='black', plot_colorbar=True,
            contour_linewidth=0.25, beam_place="lr", plot_title=source)
fig.savefig(os.path.join(data_dir, f"{source}_diff_spix_XYJU_I_X.png"), dpi=600, bbox_inches="tight")
