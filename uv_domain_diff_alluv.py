import os
import astropy.io.fits as pf
import astropy.units as un
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import sys
sys.path.insert(0, '/home/ilya/github/ve/vlbi_errors')
from uv_data import UVData
from from_fits import create_model_from_fits_file
import matplotlib
matplotlib.use("TkAgg")
label_size = 20
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size
matplotlib.rcParams['axes.titlesize'] = label_size
matplotlib.rcParams['axes.labelsize'] = label_size
matplotlib.rcParams['font.size'] = label_size
matplotlib.rcParams['legend.fontsize'] = label_size
matplotlib.rcParams["contour.negative_linestyle"] = 'dotted'

deg2rad = un.deg.to(un.rad)

PA = None
# uvclipping is done
uv_range = [1.287, 242.1]


def plot_common_uv(uvfits_u="/home/ilya/data/alpha/BK145/1228+126.U.2009_05_23C_ta60.uvf_cal",
                   uvfits_x="/home/ilya/data/alpha/BK145/1228+126.X.2009_05_23_ta60.uvf_cal",
                   uv_range=(1.287, 242.1), save_dir="/home/ilya/Documents/EVN2022"):
    uvdata_u = UVData(uvfits_u)
    uvdata_x = UVData(uvfits_x)
    uv_u = uvdata_u.uv
    r = np.hypot(uv_u[:, 0], uv_u[:, 1])
    mask_u = np.logical_or(r < uv_range[0]*1E+06, r > uv_range[1]*1E+06)

    uv_x = uvdata_x.uv
    r = np.hypot(uv_x[:, 0], uv_x[:, 1])
    mask_x = np.logical_or(r < uv_range[0]*1E+06, r > uv_range[1]*1E+06)


    rrll = uvdata_x.uvdata_freq_averaged[:, :2]
    # Masked array
    I_x = np.ma.mean(rrll, axis=1)
    weights_mask_x = I_x.mask
    I_x = I_x[~weights_mask_x]

    rrll = uvdata_u.uvdata_freq_averaged[:, :2]
    # Masked array
    I_u = np.ma.mean(rrll, axis=1)
    weights_mask_u = I_u.mask

    mask_u = np.logical_or(mask_u, weights_mask_u)
    mask_x = np.logical_or(mask_x, weights_mask_x)
    # uv_u = uv_u[mask_u]/1E+06
    uv_u = uv_u/1E+06
    # uv_x = uv_x[mask_x]/1E+06
    uv_x = uv_x[~weights_mask_x]/1E+06

    fig, axes = plt.subplots(1, 1, figsize=[6.4, 6.4])
    axes.scatter(uv_u[:, 0], uv_u[:, 1], s=1, color="C0")
    axes.scatter(-uv_u[:, 0], -uv_u[:, 1], s=1, color="C0")
    axes.set_xlabel(r"$u$, M$\lambda$")
    axes.set_ylabel(r"$v$, M$\lambda$")
    axes.set_aspect("equal")
    # axes.set_ylim([-120, 120])
    # axes.set_ylim([-250, 250])
    # axes.set_xlim([-250, 250])
    axes.set_ylim([-500, 500])
    axes.set_xlim([-500, 500])
    axes.invert_xaxis()
    plt.savefig(os.path.join(save_dir, "uv_coverage_u.png"), bbox_inches="tight", dpi=600)
    plt.show()

    fig, axes = plt.subplots(1, 1, figsize=[6.4, 6.4])
    axes.scatter(uv_x[:, 0], uv_x[:, 1], s=1, color="C0")
    axes.scatter(-uv_x[:, 0], -uv_x[:, 1], s=1, color="C0")
    axes.set_xlabel(r"$u$, M$\lambda$")
    axes.set_ylabel(r"$v$, M$\lambda$")
    axes.set_aspect("equal")
    # axes.set_ylim([-120, 120])
    # axes.set_ylim([-250, 250])
    # axes.set_xlim([-250, 250])
    axes.set_ylim([-500, 500])
    axes.set_xlim([-500, 500])
    axes.invert_xaxis()
    plt.savefig(os.path.join(save_dir, "uv_coverage_x.png"), bbox_inches="tight", dpi=600)
    plt.show()

    fig, axes = plt.subplots(1, 1, figsize=[6.4, 6.4])
    axes.scatter(np.hypot(uv_x[:, 0], uv_x[:, 1]), np.abs(I_x), s=2, color="C0")
    # axes.scatter(-uv_x[:, 0], -uv_x[:, 1], s=1, color="C0")
    axes.set_xlabel(r"$r_{uv}$, M$\lambda$")
    axes.set_ylabel(r"Vis amp, Jy")
    # axes.set_aspect("equal")
    # axes.set_ylim([-120, 120])
    # axes.set_ylim([-250, 250])
    # axes.set_xlim([-250, 250])
    # axes.set_ylim([-500, 500])
    # axes.set_xlim([-500, 500])
    # axes.invert_xaxis()
    plt.savefig(os.path.join(save_dir, "radplot_x.png"), bbox_inches="tight", dpi=600)
    plt.show()

    fig, axes = plt.subplots(1, 1, figsize=[6.4, 6.4])
    axes.scatter(np.hypot(uv_u[:, 0], uv_u[:, 1]), np.abs(I_u), s=2, color="C0")
    # axes.scatter(-uv_x[:, 0], -uv_x[:, 1], s=1, color="C0")
    axes.set_xlabel(r"$r_{uv}$, M$\lambda$")
    axes.set_ylabel(r"Vis amp, Jy")
    # axes.set_aspect("equal")
    # axes.set_ylim([-120, 120])
    # axes.set_ylim([-250, 250])
    # axes.set_xlim([-250, 250])
    # axes.set_ylim([-500, 500])
    # axes.set_xlim([-500, 500])
    # axes.invert_xaxis()
    # plt.savefig(os.path.join(save_dir, "radplot_u.png"), bbox_inches="tight", dpi=600)
    plt.show()


data_dir = "/home/ilya/data/alpha/blind_clean/BK145/uv_clipping/fix_beam_bpa/radplot"
# data_dir = "/home/ilya/data/alpha/blind_clean/BK145/uv_clipping/fix_beam_bpa/radplot/rot90"
save_dir = "/home/ilya/data/alpha/blind_clean/BK145/uv_clipping/fix_beam_bpa/radplot/alluv"
uvfits = "template_8.1.uvf"
# uvfits = "template_15.4.uvf"
cc_dfm = "model_8.1.dfm"
# cc_dfm = "model_15.4.dfm"
cc_fits = "model_cc_i_8.1.fits"
# cc_fits = "model_cc_i_15.4.fits"
model_dfm = "true_jet_model_i_8.1_rotated.txt"
model_fits = "convolved_true_jet_model_i_rotated_8.1.fits"
# model_fits = "convolved_true_jet_model_i_rotated_15.4.fits"


model_ccmodel = create_model_from_fits_file(os.path.join(data_dir, model_fits))
data_ccmodel = create_model_from_fits_file(os.path.join(data_dir, cc_fits))

uvdata = UVData(os.path.join(data_dir, uvfits))
uv = uvdata.uv.copy()
u = uv[:, 0]
v = uv[:, 1]


# Make uv-plane
u_ = np.linspace(-250E+6, 250E+6, 100)
v_ = np.linspace(-250E+6, 250E+6, 100)
u, v = np.meshgrid(u_, v_)

rrll = uvdata.uvdata_freq_averaged[:, :2]
# Masked array
I = np.ma.mean(rrll, axis=1)
weights_mask = I.mask

cc_data = pf.getdata(os.path.join(data_dir, cc_fits), extname="AIPS CC")
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

ft_cc = np.zeros(u.shape, dtype=complex)
ft_model = np.zeros(u.shape, dtype=complex)

for flux, x0, y0 in components_cc:
    ft_cc += flux*np.exp(-2.0*np.pi*1j*(u*x0 + v*y0))

for flux, x0, y0 in components_model:
    ft_model += flux*np.exp(-2.0*np.pi*1j*(u*x0 + v*y0))


y_cc = np.real(np.sqrt(ft_cc*np.conj(ft_cc)))
y_model = np.real(np.sqrt(ft_model*np.conj(ft_model)))
diff = ft_cc - ft_model
diff = np.real(np.sqrt(diff*np.conj(diff)))



from matplotlib.ticker import LogFormatter

# fig, axes = plt.subplots(1, 1, figsize=[10, 5])
fig, axes = plt.subplots(1, 1, figsize=[6.4, 6.4])
# sc = axes.pcolor(u/10**6, v/10**6, np.log10(diff/y_model), cmap="plasma", clim=[-3.7, 0.83])
sc = axes.pcolor(u/10**6, v/10**6, diff/y_model, cmap="inferno", norm=matplotlib.colors.LogNorm())#, clim=[-3.7, 0.83])
divider = make_axes_locatable(axes)
cax = divider.append_axes("right", size="3%", pad=0.00)
formatter = LogFormatter(10, labelOnlyBase=False)

cb = fig.colorbar(sc, cax=cax, format=formatter)#, ticks=[])
cb.set_label(r"Frac. error")
# cb.set_label(r"$\lg({\rmFrac. error})$")
# cb.set_label(r"$\lg({\rm Error (Jy)})$")
axes.set_xlabel(r"$u$, M$\lambda$")
axes.set_ylabel(r"$v$, M$\lambda$")
axes.set_aspect("equal")
# axes.set_ylim([-120, 120])
axes.set_ylim([-250, 250])
axes.set_xlim([-250, 250])
axes.invert_xaxis()
# plt.savefig(os.path.join(save_dir, "error_8GHz_2ridges_with_conj_uvall_250Ml_logcolorbar.png"), bbox_inches="tight", dpi=600)
plt.show()

