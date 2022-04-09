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
matplotlib.use("Qt5Agg")
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

data_dir = "/home/ilya/data/alpha/blind_clean/BK145/uv_clipping/fix_beam_bpa/radplot"
save_dir = "/home/ilya/data/alpha/blind_clean/BK145/uv_clipping/fix_beam_bpa/radplot/newpics"
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
r_uv = np.hypot(uv[:, 0], uv[:, 1])
pa_uv = np.arctan2(u, v)
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

ft_cc = np.zeros(len(uv), dtype=complex)
ft_model = np.zeros(len(uv), dtype=complex)

for flux, x0, y0 in components_cc:
    ft_cc += (flux*np.exp(-2.0*np.pi*1j*(u[:, np.newaxis]*x0 +
                                         v[:, np.newaxis]*y0))).sum(axis=1)

for flux, x0, y0 in components_model:
    ft_model += (flux*np.exp(-2.0*np.pi*1j*(u[:, np.newaxis]*x0 +
                                            v[:, np.newaxis]*y0))).sum(axis=1)



good = np.logical_and(r_uv/10**6 > uv_range[0], r_uv/10**6 < uv_range[1])
toogood = np.logical_and(good, ~weights_mask)


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
# plt.savefig(os.path.join(data_dir, "radplot_error_15GHz_2ridges.png"), bbox_inches="tight", dpi=600)
plt.show()


# fig, axes = plt.subplots(1, 1, figsize=[10, 5])
fig, axes = plt.subplots(1, 1, figsize=[6.4, 6.4])
sc = axes.scatter(u[toogood]/10**6, v[toogood]/10**6, c=np.log10(diff[toogood]/y_model[toogood]), cmap="plasma")
axes.scatter(-u[toogood]/10**6, -v[toogood]/10**6, c=np.log10(diff[toogood]/y_model[toogood]), cmap="plasma")
# sc = axes.scatter(u[toogood]/10**6, v[toogood]/10**6, c=np.log10(diff[toogood]), cmap="plasma")
divider = make_axes_locatable(axes)
cax = divider.append_axes("right", size="3%", pad=0.00)
cb = fig.colorbar(sc, cax=cax)
cb.set_label(r"$\lg({\rmFrac. error})$")
# cb.set_label(r"$\lg({\rm Error (Jy)})$")
axes.set_xlabel(r"$u$, M$\lambda$")
axes.set_ylabel(r"$v$, M$\lambda$")
axes.set_aspect("equal")
# axes.set_ylim([-120, 120])
axes.set_ylim([-250, 250])
axes.set_xlim([-250, 250])
axes.invert_xaxis()
plt.savefig(os.path.join(save_dir, "error_8GHz_2ridges_with_conj.png"), bbox_inches="tight", dpi=600)
plt.show()
