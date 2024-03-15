import os
import numpy as np
import matplotlib
# matplotlib.use("TkAgg")
import scienceplots
import matplotlib.pyplot as plt
import scienceplots
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cycler import cycler
# For tics and line widths. Re-write colors and figure size later.
plt.style.use('science')
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



results_dir = "/home/ilya/github/bk_transfer/cmake-build-debug"
tau_fr_file = "jet_image_taufr_u.txt"

image = np.loadtxt(os.path.join(results_dir, tau_fr_file))
fig, axes = plt.subplots(1, 1)
im = axes.matshow(image, cmap="bwr")
divider = make_axes_locatable(axes)
cax = divider.append_axes("right", size="5%", pad=0.00)
cb = fig.colorbar(im, cax=cax)
cb.set_label("Faraday depth")
axes.set_xlabel("along")
axes.set_ylabel("across")
fig.savefig("Faraday_depth_image.png", bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(1, 1)
axes.plot(image[:, 100])
axes.set_xlabel("across")
axes.set_ylabel("Faraday depth")
axes.set_xlim([0, 100])
fig.savefig("Faraday_depth_slice.png", bbox_inches="tight")
plt.show()

