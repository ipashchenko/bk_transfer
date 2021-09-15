import numpy as np
import os


def simulate_random_evpa(std_evpa, n_epochs=30, n_rep=1000):
    q = np.ones(n_epochs)
    u = np.ones(n_epochs)
    p = np.hypot(np.mean(q), np.mean(u))
    ratios = list()
    for i in range(n_rep):
        evpa = np.random.normal(0, std_evpa, n_epochs)
        q_ = q*np.cos(2*evpa) + u*np.sin(2*evpa)
        u_ = -q*np.sin(2*evpa) + u*np.cos(2*evpa)
        p_ = np.hypot(np.mean(q_), np.mean(u_))
        ratios.append(p_/p)
    return np.mean(ratios), np.std(ratios)


def simulate_jump_evpa(n_flips, n_epochs=30, n_rep=1000):
    q = np.ones(n_epochs)
    u = np.ones(n_epochs)
    p = np.hypot(np.mean(q), np.mean(u))
    ratios = list()
    for i in range(n_rep):
        evpa = np.zeros(n_epochs)
        evpa[:n_flips] += np.pi/2
        q_ = q*np.cos(2*evpa) + u*np.sin(2*evpa)
        u_ = -q*np.sin(2*evpa) + u*np.cos(2*evpa)
        p_ = np.hypot(np.mean(q_), np.mean(u_))
        ratios.append(p_/p)
    return np.mean(ratios), np.std(ratios)


def beta(Gamma):
    """
    Velocity in units of speed of light [c].

    :param Gamma:
        Lorentz factor.
    """
    return np.sqrt(Gamma**2.-1.)/Gamma


def delta(Gamma, theta):
    """
    Doppler factor

    :param Gamma:
        Lorentz factor.
    :param theta:
        LOS angle [rad].
    """
    return 1./(Gamma*(1.-beta(Gamma)*np.cos(theta)))


def theta_obs(theta_plasma, Gamma):
    a = 1.0
    b = Gamma*beta(Gamma)*np.sin(theta_plasma)
    c = Gamma*np.sin(theta_plasma)
    if theta_plasma <= np.pi/2:
        return -np.arccos(c/np.hypot(a, b)) + np.arctan(a/b)
    else:
        return np.arccos(c/np.hypot(a, b)) + np.arctan(a/b)


def find_Gamma_given(theta_plasmas_deg=np.arange(18, 31), theta_obs_deg=18):
    from scipy.optimize import fmin

    def func(Gamma, theta_plasma_deg):
        return np.abs(np.deg2rad(theta_obs_deg) - theta_obs(np.deg2rad(theta_plasma_deg), Gamma))

    Gammas = list()
    for theta_plasma_deg in theta_plasmas_deg:
        result = fmin(func, 2.0, args=(theta_plasma_deg,))
        Gammas.append(result[0])

    return Gammas


def theta_plasma(theta_obs, Gamma):
    return np.arcsin(np.sin(theta_obs)/(Gamma*(1-beta(Gamma)*np.cos(theta_obs))))


def sin_theta_plasma(theta_obs, Gamma):
    return np.sin(theta_obs)/(Gamma*(1-beta(Gamma)*np.cos(theta_obs)))


def cos_theta_plasma(theta_obs, Gamma):
    return (np.cos(theta_obs) - beta(Gamma))/(1-beta(Gamma)*np.cos(theta_obs))


def tb_pixel(flux, freq, size, z=0., D=1.):
    """
    Brightness temperature.

    :param flux:
        Flux in Jy.
    :param freq:
        Frequency in GHz.
    :param size:
        Size in mas.
    :return:
        Value of true brightness temperature (corrected for Doppler and
        redshift).
    """
    mas_to_rad = 4.8481368 * 1E-09
    k = 1.38 * 10 ** (-16)
    c = 3e10
    freq *= 10**9
    size *= mas_to_rad
    flux *= 10**(-23)
    Tb = c**2*flux/(2.*k*size**2*freq**2)
    return (1.+z)*Tb/D


def n_across(lg_pixel_size_mas_min, lg_pixel_size_mas_max, n_along, theta, phi):
    along_size_mas = np.sum(np.logspace(lg_pixel_size_mas_min, lg_pixel_size_mas_max, n_along))
    print("Along size = {:.1f} mas".format(along_size_mas))
    phi_app_model = np.arctan(np.tan(phi)/np.sin(theta))
    print("Apparent angle = {:.1f} deg".format(np.rad2deg(phi_app_model)))
    return 2+2*int(along_size_mas*np.tan(phi_app_model)/10**lg_pixel_size_mas_max)


def density_profile(r, r1, r2, K1, K2, b1=1.0, b2=0.0, b3=1.0):
    """
    Density profile of the (e.g. emitting from power-law or thermal) particles with central core (from r=0 to r1) and
    sheath (from r2 up to the jet border). Relative contribution of core and sheath can be changed using ``b``
    coefficients. E.g. core only corresponds to ``b1 = 1``, ``b2 = 0``, ``b3 = 0``.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(0, 1, 1000)
    >>> profile = density_profile(x, 0.1, 0.9, 30, 10, b1=1.0, b2=0.0, b3=1.0)
    >>> fig, axes = plt.subplots(1, 1)
    >>> plt.plot(x, profile)
    >>> plt.show()

    :param r:
        Single number or array with values of transverse distance (normed to the jet radius) where to calculate density
        profile.
    :param r1:
        Radius of inner core - (0, 1).
    :param r2:
        Inner radius of sheath - (0, 1)
    :param K1:
        Parameter that defines how smooth is the change from core density to zero - (1, +inf)
    :param K2:
        Parameter that defines how smooth is the change from zero to sheath density - (1, +inf)
    :param b1: (optional)
        Relative amplitude of particles density in central core. (default: ``1.0``)
    :param b2: (optional)
        Relative amplitude of particles density between sheath and core. (default: ``0.0``)
    :param b3: (optional)
        Relative amplitude of particles density in sheath. (default: ``1.0``)
    :return:
        Number or array of relative density profile, where 0 - no particles, 1 - all particles.
    """

    def linear(a, b, x):
        return a*x + b

    l1 = np.tanh(K1*(r-r1))
    l2 = np.tanh(K2*(r-r2))
    # r < r1
    y1 = linear(0.0, b1, r)
    y2 = linear(0.0, b2, r)
    y3 = linear(0.0, b3, r)
    # r < r2
    y4 = y1 + 0.5*(1+l1)*(y2-y1)
    # All r
    y5 = y4 + 0.5*(1+l2)*(y3-y4)
    return y5


def concatenate_jet_with_counter_jet(stokes, freq_ghz, path):
    j = np.loadtxt(os.path.join(path, "jet_image_{}_{}.txt".format(stokes, freq_ghz)))
    cj = np.loadtxt(os.path.join(path, "cjet_image_{}_{}.txt".format(stokes, freq_ghz)))
    jcj = np.hstack((cj[::, ::-1], j))
    np.savetxt(os.path.join(path, "jet_cjet_image_{}_{}.txt".format(stokes, freq_ghz)), jcj)