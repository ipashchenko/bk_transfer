import os
import numpy as np
from astropy import cosmology
import astropy.units as u
import matplotlib
label_size = 20
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size
matplotlib.rcParams['axes.titlesize'] = label_size
matplotlib.rcParams['axes.labelsize'] = label_size
matplotlib.rcParams['font.size'] = label_size
matplotlib.rcParams['legend.fontsize'] = label_size
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

cosmo = cosmology.WMAP9
# cm/s
c = 29979245800.0
# cm
pc = 3.0856775814671913e+18
#
day_to_sec = 86400.0


def N(r_proj_pc, t_obs_days, t_start_days, amp, l_pc, Gamma, theta_deg, N_1, n, z):
    theta = np.deg2rad(theta_deg)
    r_pc = r_proj_pc/np.sin(theta)
    beta = np.sqrt(Gamma**2 - 1)/Gamma
    beta_obs = beta/(1 - beta*np.cos(theta))/(1+z)
    return N_1*r_pc**(-n)*(1 + amp*np.exp(-(r_pc - beta_obs*c*(t_obs_days - t_start_days)*day_to_sec/pc)**2 / l_pc**2))


def B(r_proj_pc, t_obs_days, t_start_days, amp, l_pc, Gamma, theta_deg, B_1, b, z):
    theta = np.deg2rad(theta_deg)
    r_pc = r_proj_pc/np.sin(theta)
    beta = np.sqrt(Gamma**2 - 1)/Gamma
    beta_obs = beta/(1 - beta*np.cos(theta))/(1+z)
    return B_1*r_pc**(-b)*(1 + amp*np.exp(-(r_pc - beta_obs*c*(t_obs_days - t_start_days)*day_to_sec/pc)**2 / l_pc**2))


def ang_to_dist(z):
    return cosmo.kpc_proper_per_arcmin(z)


def mas_to_pc(mas_coordinates, z):
    return (mas_coordinates * u.mas * ang_to_dist(z)).to(u.pc).value


def get_proj_core_position(image_txt, tau_txt, z, lg_pixel_size_mas_min, lg_pixel_size_mas_max, n_along, n_across):
    resolutions = np.logspace(lg_pixel_size_mas_min, lg_pixel_size_mas_max, n_along)
    pixsize_array = np.tile(resolutions, n_across).reshape(n_across, n_along).T
    intensity_factor = (pixsize_array/np.min(pixsize_array))**2
    pixel_coordinates = np.cumsum(resolutions) - resolutions/2

    tau = np.loadtxt(tau_txt)
    I_orig = np.loadtxt(image_txt)
    I = I_orig/intensity_factor.T
    midle = int(n_across/2)
    tau_stripe = 0.5*(tau[midle-1, :] + tau[midle, :])
    I_stripe = 0.5*(I[midle-1, :] + I[midle, :])
    I_mean = np.sum(I, axis=0)

    idx_max_I_stripe = np.argmax(I_stripe)
    idx_max_I_mean = np.argmax(I_mean)

    # FIXME: If several tau = 1 region are present - get the first close to 1
    # First one is jet apex
    idx_tau_1 = np.where(tau_stripe < 1)[0][0]
    if idx_tau_1 == 0:
        idx_tau_1 = np.where(tau_stripe < 1)[0][1]
    if idx_tau_1 == 1:
        idx_tau_1 = np.where(tau_stripe < 1)[0][2]
    if idx_tau_1 == 2:
        idx_tau_1 = np.where(tau_stripe < 1)[0][3]
    if idx_tau_1 == 3:
        idx_tau_1 = np.where(tau_stripe < 1)[0][4]

    # Find if there is another tau = 1 region due to flare down the core
    tau_down_core = tau_stripe[idx_tau_1:]
    try:
        idx_tau_1_2 = np.where(tau_down_core > 1)[0][-1]
        # Index in full size
        idx_tau_1_2 = np.arange(len(tau_stripe))[idx_tau_1:][idx_tau_1_2]
    # No second tau = 1 region
    except IndexError:
        print("No second tau = 1 region")
        idx_tau_1_2 = None

    tau_at_max_I_stripe = tau_stripe[idx_max_I_stripe]
    tau_at_max_I_mean = tau_stripe[idx_max_I_mean]

    print("Tau at max I stripe = ", tau_at_max_I_stripe)
    print("Tau at max I mean = ", tau_at_max_I_mean)

    pos_tau_1_1_mas = pixel_coordinates[idx_tau_1]
    pos_tau_1_1_pc = mas_to_pc(pos_tau_1_1_mas, z)
    pos_tau_1_2_mas = None
    pos_tau_1_2_pc = None
    if idx_tau_1_2 is not None:
        pos_tau_1_2_mas = pixel_coordinates[idx_tau_1_2]
        pos_tau_1_2_pc = mas_to_pc(pos_tau_1_2_mas, z)
    pos_max_I_stripe_mas = pixel_coordinates[idx_max_I_stripe]
    pos_max_I_stripe_pc = mas_to_pc(pos_max_I_stripe_mas, z)
    pos_max_I_mean_mas = pixel_coordinates[idx_max_I_mean]
    pos_max_I_mean_pc = mas_to_pc(pos_max_I_mean_mas, z)

    print("Shape = ", I_orig.shape)
    print("idx_tau_1 = ", idx_tau_1)
    core_flux = np.sum(I_orig[:, :idx_tau_1])
    jet_flux = np.sum(I_orig[:, idx_tau_1:])


    return {"tau_1_1": pos_tau_1_1_pc, "tau_1_2": pos_tau_1_2_pc,
            "max_I_stripe": pos_max_I_stripe_pc, "max_I_mean": pos_max_I_mean_pc,
            "core_flux": core_flux, "tau_1_1_mas": pos_tau_1_1_mas, "tau_1_2_mas": pos_tau_1_2_mas,
            "max_I_stripe_mas": pos_max_I_stripe_mas, "max_I_mean_mas": pos_max_I_mean_mas,
            "core_flux": core_flux, "jet_flux": jet_flux}



# z = 0.5
# freq_ghz = 2.1
# lg_pixel_size_mas_min = -2.0
# lg_pixel_size_mas_max = -1.0
# n_along = 1024
# n_across = 512
# data_dir = "/home/ilya/github/bk_transfer/Release"
#
# resolutions = np.logspace(lg_pixel_size_mas_min, lg_pixel_size_mas_max, n_along)
# pixsize_array = np.tile(resolutions, n_across).reshape(n_across, n_along).T
# intensity_factor = (pixsize_array/np.min(pixsize_array))**2
# pixel_coordinates = np.cumsum(resolutions) - resolutions/2
#
#
# tau = np.loadtxt(os.path.join(data_dir, "jet_image_tau_{}.txt".format(freq_ghz)))
# I = np.loadtxt(os.path.join(data_dir, "jet_image_i_{}.txt".format(freq_ghz)))/intensity_factor.T
# midle = int(n_across/2)
# tau_stripe = 0.5*(tau[midle-1, :] + tau[midle, :])
# I_stripe = 0.5*(I[midle-1, :] + I[midle, :])
# I_mean = np.sum(I, axis=0)
#
# idx_max_I_stripe = np.argmax(I_stripe)
# idx_max_I_mean = np.argmax(I_mean)
#
# # FIXME: If several tau = 1 region are present - get the first close to 1
# # First one is jet apex
# idx_tau_1 = np.where(tau_stripe < 1)[0][0]
# if idx_tau_1 == 0:
#     idx_tau_1 = np.where(tau_stripe < 1)[0][1]
# # Find if there is another tau = 1 region due to flare down the core
# tau_down_core = tau_stripe[idx_tau_1:]
# try:
#     idx_tau_1_2 = np.where(tau_down_core > 1)[0][-1]
#     # Index in full size
#     idx_tau_1_2 = np.arange(len(tau_stripe))[idx_tau_1:][idx_tau_1_2]
# # No second tau = 1 region
# except IndexError:
#     print("No second tau = 1 region")
#     idx_tau_1_2 = None
#
# tau_at_max_I_stripe = tau_stripe[idx_max_I_stripe]
# tau_at_max_I_mean = tau_stripe[idx_max_I_mean]
#
# print("Tau at max I stripe = ", tau_at_max_I_stripe)
# print("Tau at max I mean = ", tau_at_max_I_mean)
#
# fig, axes = plt.subplots(1, 1)
# axes_I = axes.twinx()
# axes.plot(pixel_coordinates, tau_stripe, '.', color="C0")
# axes.axvline(pixel_coordinates[idx_tau_1], color="C0", label=r"$\tau = 1$")
# axes_I.plot(pixel_coordinates, I_stripe, label="stripe", color="C1")
# axes_I.plot(pixel_coordinates, I_mean, "--", label="sum", color="C1")
# axes_I.axvline(pixel_coordinates[idx_max_I_mean], color="C1", label=r"$\tau = {:.2f}$".format(tau_at_max_I_mean))
# axes.set_yscale('log')
# axes_I.set_yscale('log')
# axes.set_xlabel("Distance, mas")
# axes.set_ylabel(r"$\tau$")
# axes_I.set_ylabel(r"$I$")
#
# lines, labels = axes.get_legend_handles_labels()
# lines2, labels2 = axes_I.get_legend_handles_labels()
# axes.legend(lines + lines2, labels + labels2, loc=0)
#
# # axes.legend(loc=0)
# # axes_I.legend(loc=1)
# plt.show()


if __name__ == "__main__":
    data_dir = "/home/ilya/data/rfc"
    # txt_dir = "/home/ilya/fs/sshfs/calculon/github/bk_transfer/Release"
    txt_dir = "/home/ilya/github/bk_transfer/Release"
    save_dir = "/home/ilya/github/bk_transfer/pics"
    source_template = "J0102+5824"
    z = 1.0
    plot = True
    lg_pixel_size_mas_min = -2.5
    lg_pixel_size_mas_max = -0.5
    n_along = 400
    n_across = 80

    t_start_days = 0.0
    amp_N = 5.0
    amp_B = 0.0
    l_pc = 0.2
    Gamma = 8.6
    theta_deg = 5.0
    b = 1.0
    B_1 = 0.85
    n = 2.0
    N_1 = 500.0
    z = 1.0
    # ts_obs_days = np.loadtxt(os.path.join(data_dir, "{}_times.txt".format(source_template)))
    ts_obs_days = np.linspace(0, 10*360, 30)
    corex_positions = list()
    cores_positions = list()
    corex_fluxes = list()
    cores_fluxes = list()
    jetx_fluxes = list()
    jets_fluxes = list()
    B_core_S = list()
    B_core_X = list()
    N_core_S = list()
    N_core_X = list()
    for t_obs_days in ts_obs_days:
        print("T[days] = {:.1f}".format(t_obs_days))
        imagex_txt = os.path.join(txt_dir, "jet_image_i_X_{:.1f}.txt".format(t_obs_days))
        images_txt = os.path.join(txt_dir, "jet_image_i_S_{:.1f}.txt".format(t_obs_days))

        # convert -delay 10 -loop 0 `ls -tr tobs*.png` animation.gif
        if plot:
            fig, axes = plt.subplots(2, 1, sharex=True)
            imagex = np.loadtxt(imagex_txt)
            images = np.loadtxt(images_txt)
            print("Total flux S = {:.2f} Jy".format(np.sum(images)))
            print("Total flux X = {:.2f} Jy".format(np.sum(imagex)))
            axes[0].matshow(np.log10(images))
            axes[1].matshow(np.log10(imagex))
            axes[1].xaxis.tick_bottom()
            axes[0].xaxis.tick_bottom()
            axes[1].set_xlabel("Along")
            fig.subplots_adjust(hspace=0)
            fig.subplots_adjust(wspace=0)
            fig.tight_layout()
            plt.savefig("tobs_{:.1f}.png".format(t_obs_days))
            plt.close()
            # plt.show()
        taux_txt = os.path.join(txt_dir, "jet_image_tau_X_{:.1f}.txt".format(t_obs_days))
        taus_txt = os.path.join(txt_dir, "jet_image_tau_S_{:.1f}.txt".format(t_obs_days))
        resx = get_proj_core_position(imagex_txt, taux_txt, z, lg_pixel_size_mas_min, lg_pixel_size_mas_max,
                                      n_along, n_across)
        ress = get_proj_core_position(images_txt, taus_txt, z, lg_pixel_size_mas_min, lg_pixel_size_mas_max,
                                      n_along, n_across)
        corex_positions.append(resx["tau_1_1_mas"])
        cores_positions.append(ress["tau_1_1_mas"])
        corex_fluxes.append(resx["core_flux"])
        cores_fluxes.append(ress["core_flux"])
        jetx_fluxes.append(resx["jet_flux"])
        jets_fluxes.append(ress["jet_flux"])
        b_core_S = B(ress["tau_1_1"], t_obs_days, t_start_days, amp_B, l_pc, Gamma, theta_deg, B_1, b, z)
        n_core_S = N(ress["tau_1_1"], t_obs_days, t_start_days, amp_N, l_pc, Gamma, theta_deg, N_1, n, z)
        b_core_X = B(resx["tau_1_1"], t_obs_days, t_start_days, amp_B, l_pc, Gamma, theta_deg, B_1, b, z)
        n_core_X = N(resx["tau_1_1"], t_obs_days, t_start_days, amp_N, l_pc, Gamma, theta_deg, N_1, n, z)
        B_core_S.append(b_core_S)
        N_core_S.append(n_core_S)
        B_core_X.append(b_core_X)
        N_core_X.append(n_core_X)
        print("Core flux S = {:.2f}".format(cores_fluxes[-1]))
        print("Jet flux S = {:.2f}".format(jets_fluxes[-1]))

    CS = np.array(cores_positions)-np.array(corex_positions)


    fig, axes = plt.subplots(1, 1, figsize=(15, 15))
    axes.set_xlabel("Time, days")
    axes2 = axes.twinx()
    axes2.set_ylabel("Flux density, Jy")
    axes.set_ylabel("Core shift, mas")
    axes.tick_params("y")
    axes2.tick_params("y")

    axes.plot([], [], color="C0", label=r"$S_{\rm core,8 GHz}$")
    axes.plot([], [], color="C1", label=r"$S_{\rm core,2 GHz}$")

    axes.plot(ts_obs_days, CS, "--", label="CS", color="black")
    axes.scatter(ts_obs_days, CS, color="black")

    axes.plot(ts_obs_days, corex_positions, "--", label=r"$r_{\rm 8 GHz}$", color="C0")
    axes.scatter(ts_obs_days, corex_positions, color="C0")
    axes.plot(ts_obs_days, cores_positions, "--", label=r"$r_{\rm 2 GHz}$", color="C1")
    axes.scatter(ts_obs_days, cores_positions, color="C1")

    axes2.plot(ts_obs_days, corex_fluxes, color="C0")
    axes2.scatter(ts_obs_days, corex_fluxes, color="C0")
    axes2.plot(ts_obs_days, cores_fluxes, color="C1")
    axes2.scatter(ts_obs_days, cores_fluxes, color="C1")

    axes.legend()
    # fig.savefig(os.path.join(save_dir, "CS_rc_Sc_t_true.png"), bbox_inches="tight")
    plt.show()


    fig, axes = plt.subplots(1, 1, figsize=(15, 15))
    axes.set_xlabel("Time, days")
    axes2 = axes.twinx()
    axes2.set_ylabel("Flux density, Jy")
    axes.set_ylabel("Core position, mas")
    axes.tick_params("y")
    axes2.tick_params("y")

    axes.plot([], [], color="C0", label=r"$S_{\rm core,8 GHz}$")
    axes.plot([], [], color="C1", label=r"$S_{\rm core,2 GHz}$")

    axes.plot(ts_obs_days, corex_positions, "--", label=r"$r_{\rm 8 GHz}$", color="C0")
    axes.scatter(ts_obs_days, corex_positions, color="C0")
    axes.plot(ts_obs_days, cores_positions, "--", label=r"$r_{\rm 2 GHz}$", color="C1")
    axes.scatter(ts_obs_days, cores_positions, color="C1")
    axes2.plot(ts_obs_days, corex_fluxes, color="C0")
    axes2.scatter(ts_obs_days, corex_fluxes, color="C0")
    axes2.plot(ts_obs_days, cores_fluxes, color="C1")
    axes2.scatter(ts_obs_days, cores_fluxes, color="C1")

    axes.legend()
    # fig.savefig(os.path.join(save_dir, "rc_Sc_t_true.png"), bbox_inches="tight")
    plt.show()


    fig, axes = plt.subplots(1, 1)
    axes.scatter(cores_fluxes, cores_positions)
    axes.set_xlabel(r"$S_{\rm core}$, Jy")
    axes.set_ylabel(r"$r_{\rm core}$, mas")
    fig.savefig(os.path.join(save_dir, "rc_Sc_true_Sband.png"), bbox_inches="tight")
    plt.show()
    fig, axes = plt.subplots(1, 1)
    axes.scatter(corex_fluxes, corex_positions)
    axes.set_xlabel(r"$S_{\rm core}$, Jy")
    axes.set_ylabel(r"$r_{\rm core}$, mas")
    # fig.savefig(os.path.join(save_dir, "rc_Sc_true_Xband.png"), bbox_inches="tight")
    plt.show()

    med_B = np.median(B_core_S)
    med_N = np.median(N_core_S)
    fig, axes = plt.subplots(1, 1)
    axes.scatter(N_core_S, B_core_S, label="flare")
    axes.scatter(med_N, med_B, s=20, color="C1", label="stationary")
    axes.set_xlabel(r"$N_{\rm core}, {\rm cm}^{-3}$")
    axes.set_ylabel(r"$B_{\rm core}, {\rm G}$")
    plt.legend()
    # fig.savefig(os.path.join(save_dir, "Bc_Nc_true_Sband.png"), bbox_inches="tight")
    plt.show()


    med_B = np.median(B_core_X)
    med_N = np.median(N_core_X)
    fig, axes = plt.subplots(1, 1)
    axes.scatter(N_core_X, B_core_X, label="flare")
    axes.scatter(med_N, med_B, s=20, color="C1", label="stationary")
    axes.set_xlabel(r"$N_{\rm core}, {\rm cm}^{-3}$")
    axes.set_ylabel(r"$B_{\rm core}, {\rm G}$")
    plt.legend()
    # fig.savefig(os.path.join(save_dir, "Bc_Nc_true_Xband.png"), bbox_inches="tight")
    plt.show()
