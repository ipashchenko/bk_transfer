import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import math
from astropy.convolution import convolve, Gaussian1DKernel


def plot(contours=None, colors=None, vectors=None, vectors_values=None,
         cmap='gist_rainbow', abs_levels=None, rel_levels=None, min_abs_level=None,
         min_rel_level=None, k=2, vinc=2, contours_mask=None, colors_mask=None,
         vectors_mask=None, color_clim=None, outfile=None, outdir=None, close=False,
         colorbar_label=None, show=True, contour_color='k', vector_color="k", plot_colorbar=True,
         max_vector_value_length=5., mas_in_pixel=None, vector_enlarge_factor=1.0,
         label_size=14, figsize=(20, 5)):
    """
    :param contours: (optional)
        Numpy 2D array (possibly masked) that should be plotted using contours.
    :param colors: (optional)
        Numpy 2D array (possibly masked) that should be plotted using colors.
    :param vectors: (optional)
        Numpy 2D array (possibly masked) that should be plotted using vectors.
    :param vectors_values: (optional)
        Numpy 2D array (possibly masked) that should be used as vector's lengths
        when plotting ``vectors`` array.
    :param cmap: (optional)
        Colormap to use for plotting colors.
        (default: ``gist_rainbow``)
    :param abs_levels: (optional)
        Iterable of absolute levels. If ``None`` then construct levels in other
        way. (default: ``None``)
    :param min_abs_level: (optional)
        Values of minimal absolute level. Used with conjunction of ``k``
        argument for building sequence of absolute levels. If ``None`` then
        construct levels in other way. (default: ``None``)
    :param rel_levels: (optional)
        Iterable of relative levels. If ``None`` then construct levels in other
        way. (default: ``None``)
    :param min_rel_level: (optional)
        Values of minimal relative level. Used with conjunction of ``k``
        argument for building sequence of relative levels. If ``None`` then
        construct levels in other way. (default: ``None``)
    :param k: (optional)
        Factor of incrementation for levels. (default: ``2.0``)
    :param colorbar_label: (optional)
        String to label colorbar. If ``None`` then don't label. (default:
        ``None``)
    :param plot_colorbar: (optional)
        If colors is set then should we plot colorbar? (default: ``True``).
    :param max_vector_value_length: (optional)
        Determines what part of the image is the length of the vector with
        maximum magnitude. E.g. if ``5`` then maximum value of vector quantity
        corresponds to arrow with length equal to 1/5 of the image length.
        (default: ``5``)
    :param mas_in_pixel: (optonal)
        Number of milliarcseconds in one pixel. If ``None`` then plot in pixels.
        (default: ``None``)
    :param vector_enlarge_factor: (optional)
        Additional factor to increase length of vectors representing direction and values of linear polarization.
    """
    matplotlib.rcParams['xtick.labelsize'] = label_size
    matplotlib.rcParams['ytick.labelsize'] = label_size
    matplotlib.rcParams['axes.titlesize'] = label_size
    matplotlib.rcParams['axes.labelsize'] = label_size
    matplotlib.rcParams['font.size'] = label_size
    matplotlib.rcParams['legend.fontsize'] = label_size
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    image = None
    if contours is not None:
        image = contours
    elif colors is not None and image is None:
        image = colors
    elif vectors is not None and image is None:
        image = vectors
    else:
        raise Exception("No image to plot")

    x = np.arange(image.shape[0]) - image.shape[0]/2
    y = np.arange(image.shape[1]) - image.shape[1]/2
    if mas_in_pixel is not None:
        x *= mas_in_pixel
        y *= mas_in_pixel

    # Optionally mask arrays
    if contours is not None and contours_mask is not None:
        contours = np.ma.array(contours, mask=contours_mask)
    if colors is not None and colors_mask is not None:
        colors = np.ma.array(colors, mask=colors_mask)
    if vectors is not None and vectors_mask is not None:
        vectors = np.ma.array(vectors, mask=vectors_mask)

    # Actually plotting
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    # Plot contours
    if contours is not None:
        if abs_levels is None:
            max_level = contours.max()
            if rel_levels is not None:
                abs_levels = [-max_level] + [max_level * i for i in rel_levels]
            else:
                if min_abs_level is not None:
                    n_max = int(math.ceil(math.log(max_level / min_abs_level, k)))
                elif min_rel_level is not None:
                    min_abs_level = min_rel_level * max_level / 100.
                    n_max = int(math.ceil(math.log(max_level / min_abs_level, k)))
                else:
                    raise Exception("Not enough information for levels")
                abs_levels = [-min_abs_level] + [min_abs_level * k ** i for i in
                                                 range(n_max)]
        co = ax.contour(y, x, contours, abs_levels, colors=contour_color)
    if colors is not None:
        im = ax.imshow(colors, interpolation='none',
                       origin='lower', extent=[y[0], y[-1], x[0], x[-1]],
                       cmap=plt.get_cmap(cmap), clim=color_clim)
    if vectors is not None:
        if vectors_values is not None:
            u = vectors_values * np.cos(vectors)
            v = vectors_values * np.sin(vectors)
            max_vector_value = np.max(np.abs(vectors_values))
            scale = max_vector_value_length*max_vector_value/vector_enlarge_factor
        else:
            u = np.cos(vectors)
            v = np.sin(vectors)
            scale = None

        if vectors_mask is not None:
            u = np.ma.array(u, mask=vectors_mask)
            v = np.ma.array(v, mask=vectors_mask)

        vec = ax.quiver(y[::vinc], x[::vinc], u[::vinc, ::vinc],
                        v[::vinc, ::vinc], angles='uv',
                        units='width', headwidth=0., headlength=0., scale=scale,
                        width=0.0025, headaxislength=0., pivot='middle',
                        scale_units='width', color=vector_color)

    # Set equal aspect
    ax.set_aspect('equal')

    if colors is not None:
        if plot_colorbar:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.00)
            cb = fig.colorbar(im, cax=cax)
            if colorbar_label is not None:
                cb.set_label(colorbar_label)

    # Saving output
    if outfile:
        if outdir is None:
            outdir = '.'
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        path = os.path.join(outdir, outfile)
        plt.savefig("{}.png".format(path), bbox_inches='tight', dpi=300)

    if show:
        plt.ioff()
        plt.show()
    if close:
        plt.close()

    return fig


if __name__ == '__main__':
    matplotlib.use("Agg")
    freq_ghz = 15.4
    freq_ghz_high = 15.4
    freq_ghz_low = 8.1

    save_dir = "/home/ilya/github/bk_transfer/pics/KH"
    txt_dir = "/home/ilya/github/bk_transfer/Release"

    i_image = np.loadtxt(os.path.join(txt_dir, 'jet_image_i_{}.txt'.format(freq_ghz_high)))
    # i_image_high = np.loadtxt(os.path.join(txt_dir, 'jet_image_i_{}.txt'.format(freq_ghz_high)))
    # i_image_low = np.loadtxt(os.path.join(txt_dir, 'jet_image_i_{}.txt'.format(freq_ghz_low)))

    # pixel_size = 0.01
    # sigma = 1.93/pixel_size/2.355
    # gauss_kernel = Gaussian1DKernel(sigma)
    # i_slice_high_conv = convolve(i_image_high[:, 200], gauss_kernel)
    # i_slice_low_conv = convolve(i_image_low[:, 200], gauss_kernel)
    # alpha_slice_conv = np.log(i_slice_low_conv/i_slice_high_conv)/np.log(freq_ghz_low/freq_ghz_high)
    # plt.plot(alpha_slice_conv)
    # plt.show()

    # import sys
    # sys.exit(0)

    q_image = np.loadtxt(os.path.join(txt_dir, 'jet_image_q_{}.txt'.format(freq_ghz)))
    u_image = np.loadtxt(os.path.join(txt_dir, 'jet_image_u_{}.txt'.format(freq_ghz)))
    v_image = np.loadtxt(os.path.join(txt_dir, 'jet_image_v_{}.txt'.format(freq_ghz)))
    tau_image = np.loadtxt(os.path.join(txt_dir, 'jet_image_tau_{}.txt'.format(freq_ghz)))
    tau_fr_image = np.loadtxt(os.path.join(txt_dir, 'jet_image_taufr_{}.txt'.format(freq_ghz)))
    # tau_fr_image_low = np.loadtxt('jet_image_taufr_{}.txt'.format(freq_ghz_low))
    # tau_fr_image_high = np.loadtxt('jet_image_taufr_{}.txt'.format(freq_ghz_high))
    rm_image = 0.5*tau_fr_image/(0.02**2)
    l_image = np.loadtxt(os.path.join(txt_dir, 'jet_image_l_{}.txt'.format(freq_ghz)))
    p_image = np.sqrt(q_image**2+u_image**2)
    fpol_image = p_image/i_image
    # alpha_image = np.log(i_image_low/i_image_high)/np.log(freq_ghz_low/freq_ghz_high)
    chi_image = 0.5*np.arctan2(u_image, q_image) - 0.5*tau_fr_image

    print("Flux density (Jy) = ", i_image.sum())
    # print("Flux density (Jy) = ", i_image_sim.sum())
    # # Plotting transverse slices with I, P and EVPA like in Murphy+2013
    # fig, axes = plt.subplots(1, 1)
    # axes.set_title(r"$\gamma^{'} = 10^{\circ}$, $\delta = 1.7^{\circ}$, $\Gamma = 3$")
    # i_max = np.max(i_image[:, 500])
    # x = np.arange(i_image.shape[0])
    # angle = 2*np.abs(chi_image[:, 500])[::-1]
    # axes.plot(x, i_image[:, 500][::-1]/i_max, label=r"$I$")
    # axes.plot(x, (p_image[:, 500][::-1]/i_max*np.cos(angle)), color="k", ls="--", label=r"$P\cos{\chi}$")
    # plt.legend()
    # axes.axhline(0, color="k")
    # axes.get_xaxis().set_ticks([])
    # axes.get_yaxis().set_ticks([])
    # axes.axvline(400, lw=0.5, color="k")
    # axes.set_xlabel("Jet cross section")
    # axes.set_ylabel("Intensity")
    # axes_left = axes.twinx()
    # axes_left.set_ylim(axes.get_ylim())
    # axes_left.get_yaxis().set_ticks([])
    # axes_left.set_ylabel("Across the jet - Along the jet")
    # plt.show()

    # colors_mask = i_image_high < i_image_high.max()*0.0000001
    # fig = plot(contours=i_image_high, colors=np.log(i_image_high), cmap="jet", colors_mask=colors_mask, min_rel_level=0.05,
    #            colorbar_label=r'$\log{I}$')
    # fig.savefig("I.png", dpi=300, bbox_inches="tight")
    # fig = plot(contours=i_image_high, colors=np.log10(tau_image), cmap="jet", colors_mask=colors_mask, min_rel_level=0.05,
    #            colorbar_label=r'$\lg{\tau}$')
    # fig.savefig("tau_15GHz.png", dpi=300, bbox_inches="tight")
    # fig = plot(contours=i_image_high, colors=alpha_image, colors_mask=colors_mask, min_rel_level=0.05,
    #            colorbar_label=r'$\alpha$', color_clim=[-0.8, 0.2])
    # fig.savefig("alpha_Icontours.png", dpi=300, bbox_inches="tight")
    # import sys
    # sys.exit(0)

    # Just plotting picture
    colors_mask = i_image < i_image.max()*0.000001
    fig = plot(contours=i_image, colors=fpol_image, vectors=chi_image,
         vectors_values=None, colors_mask=colors_mask, min_rel_level=0.01,
         vinc=10, vectors_mask=colors_mask, vector_color="k", contour_color="k", cmap="jet", color_clim=[0, 0.75],
         colorbar_label='Frac. LP', vector_enlarge_factor=8)#, outdir="/home/ilya", outfile="oldMHDsimulations_LinPol_NUpixel.png")
    fig.savefig(os.path.join(save_dir, "LinPol_Icontours.png"), dpi=300, bbox_inches="tight")

    fig = plot(contours=p_image, colors=p_image, vectors=chi_image,
               vectors_values=None, colors_mask=colors_mask, min_rel_level=0.01,
               vinc=10, vectors_mask=colors_mask, contour_color="k", vector_color="k", cmap="gist_rainbow",
               vector_enlarge_factor=8, colorbar_label="P")#, outdir="/home/ilya", outfile="oldMHDsimulations_LinPol_NUpixel.png")
    fig.savefig(os.path.join(save_dir, "LinPol_Pcontours.png"), dpi=300, bbox_inches="tight")

    fig = plot(contours=i_image, colors=p_image, cmap="jet", colors_mask=colors_mask, min_rel_level=0.05,
               colorbar_label=r'$P$')
    fig.savefig(os.path.join(save_dir, "P.png"), dpi=300, bbox_inches="tight")

    fig = plot(contours=p_image, colors=fpol_image, colors_mask=colors_mask, min_rel_level=0.05, color_clim=[0, 0.75],
         colorbar_label=r'$FPOL$', cmap="jet")
    fig.savefig(os.path.join(save_dir, "P_FPOL.png"), dpi=300, bbox_inches="tight")

    max_tau_fr = np.max(np.abs(tau_fr_image))
    fig = plot(contours=i_image, colors=tau_fr_image, colors_mask=colors_mask, min_rel_level=0.05,
               colorbar_label=r'$\tau_{\rm FR}$', cmap="bwr", color_clim=[-max_tau_fr, max_tau_fr])#, outdir="/home/ilya", outfile="oldMHDsimulations_LinPol_NUpixel.png")
    fig.savefig(os.path.join(save_dir, "FaradayDepth_15GHz.png"), dpi=300, bbox_inches="tight")

    max_rm = np.max(np.abs(rm_image))
    fig = plot(contours=i_image, colors=rm_image, colors_mask=colors_mask, min_rel_level=0.05,
               colorbar_label=r'RM, rad/m$^2$', color_clim=[-max_rm, max_rm], cmap="bwr")
    fig.savefig(os.path.join(save_dir, "RM.png"), dpi=300, bbox_inches="tight")
