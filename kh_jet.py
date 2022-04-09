from abc import ABC
import functools
import numpy as np
import ehtim as eh
import pandas as pd
pd.options.mode.chained_assignment = None
from astropy import units as u, cosmology
import dlib
import pybobyqa
import sys
sys.path.insert(0, '/home/ilya/github/ve/vlbi_errors')
from uv_data import UVData, downscale_uvdata_by_freq
from from_fits import create_clean_image_from_fits_file
from image import plot as iplot
from spydiff import clean_difmap
try:
    from fourier import FINUFFT_NUNU
except ImportError:
    raise Exception("Install pynufft")
    FINUFFT_NUNU = None
from pykh import run_on_analytic_params_kh
from vlbi_utils import (find_image_std, find_bbox, convert_difmap_model_file_to_CCFITS, rotate_difmap_model)


def make_datafiles(uvfits, camp_txt, cph_txt, uv_txt, avg_time_logcamp=60, avg_time_cph=60, snr_cut_logcamp=0., snr_cut_cph=0.):
    # Fitting only first IF
    # obs = eh.obsdata.load_uvfits(uvfits, IF=[0])
    obs = eh.obsdata.load_uvfits(uvfits)

    # factor = obs.estimate_noise_rescale_factor()
    # obs = obs.rescale_noise(factor)

    print(obs.tarr)
    # Sort array by median SNR!
    obs.reorder_tarr_snr()
    print(obs.tarr)


    # # Old
    # camp = obs.c_amplitudes(debias=True, ctype="logcamp")
    # camp_df = pd.DataFrame.from_records(camp)

    # New
    # Adds attribute self.logcamp
    obs.add_logcamp(avg_time=avg_time_logcamp, return_type='df', ctype='logcamp',
                    count='min', debias=True, snrcut=snr_cut_logcamp,
                    err_type='predicted', num_samples=1000, round_s=0.1)
    camp_df = obs.logcamp[["time", "t1", "t2", "t3", "t4", "u1", "v1", "u2", "v2", "u3", "v3", "u4", "v4", "camp", "sigmaca"]]


    camp_uv = pd.DataFrame(np.vstack((camp_df[["u1", "v1"]].values,
                                      camp_df[["u2", "v2"]].values,
                                      camp_df[["u3", "v3"]].values,
                                      camp_df[["u4", "v4"]].values)), columns=["u", "v"])
    # Adds attribute self.cphase
    obs.add_cphase(avg_time=avg_time_cph, return_type='df', count='min', snrcut=snr_cut_cph,
                   err_type='predicted', num_samples=1000, round_s=0.1, uv_min=False)
    cph_df = obs.cphase[["time", "t1", "t2", "t3", "u1", "v1", "u2", "v2", "u3", "v3", "cphase", "sigmacp"]]
    cph_df["cphase"] = np.deg2rad(cph_df["cphase"])
    cph_df["sigmacp"] = np.deg2rad(cph_df["sigmacp"])
    cph_uv = pd.DataFrame(np.vstack((cph_df[["u1", "v1"]].values,
                                     cph_df[["u2", "v2"]].values,
                                     cph_df[["u3", "v3"]].values)), columns=["u", "v"])

    uv_df = pd.concat([camp_uv, cph_uv])
    uv_df = uv_df.drop_duplicates().reset_index(drop=True)

    # For each camp & cphase find vectors (camp_indx1, .., camp_indx4) and (cphase_indx1, ..., cphase_indx3) with indices
    # of (u1, v1), (u2, v2) in common uv vector.
    camp_indx1 = list()
    camp_indx2 = list()
    camp_indx3 = list()
    camp_indx4 = list()
    cphase_indx1 = list()
    cphase_indx2 = list()
    cphase_indx3 = list()
    for _, row in camp_df.iterrows():
        camp_indx1.append(uv_df.index[np.logical_and(uv_df['u'] == row["u1"], uv_df["v"] == row["v1"])][0])
        camp_indx2.append(uv_df.index[np.logical_and(uv_df['u'] == row["u2"], uv_df["v"] == row["v2"])][0])
        camp_indx3.append(uv_df.index[np.logical_and(uv_df['u'] == row["u3"], uv_df["v"] == row["v3"])][0])
        camp_indx4.append(uv_df.index[np.logical_and(uv_df['u'] == row["u4"], uv_df["v"] == row["v4"])][0])
    for _, row in cph_df.iterrows():
        cphase_indx1.append(uv_df.index[np.logical_and(uv_df['u'] == row["u1"], uv_df["v"] == row["v1"])][0])
        cphase_indx2.append(uv_df.index[np.logical_and(uv_df['u'] == row["u2"], uv_df["v"] == row["v2"])][0])
        cphase_indx3.append(uv_df.index[np.logical_and(uv_df['u'] == row["u3"], uv_df["v"] == row["v3"])][0])

    camp_df["indx1"] = camp_indx1
    camp_df["indx2"] = camp_indx2
    camp_df["indx3"] = camp_indx3
    camp_df["indx4"] = camp_indx4
    cph_df["indx1"] = cphase_indx1
    cph_df["indx2"] = cphase_indx2
    cph_df["indx3"] = cphase_indx3
    camp_df = camp_df[["camp", "sigmaca", "indx1", "indx2", "indx3", "indx4"]]
    cph_df = cph_df[["cphase", "sigmacp", "indx1", "indx2", "indx3"]]
    cph_df.to_csv(cph_txt, sep=" ", header=False, index=False)
    camp_df.to_csv(camp_txt, sep=" ", header=False, index=False)
    uv_df.to_csv(uv_txt, sep=" ", header=False, index=False)


class JetImage(ABC):
    cosmo = cosmology.WMAP9
    """
        ``rot=0`` corresponds to South. ``rot>0`` means rotation clock-wise from
        South.
        ''dx>0`` shifts to negative RA.
        ''dy>0`` shifts to negative DEC.
    """
    def __init__(self, z, n_along, n_across, lg_pixel_size_mas_min, lg_pixel_size_mas_max, ft_class=FINUFFT_NUNU,
                 jet_side=True, dx=0.0, dy=0.0, rot=0.0, stokes="I"):
        self.jet_side = jet_side
        self.z = z
        self.lg_pixel_size_mas_min = lg_pixel_size_mas_min
        self.lg_pixel_size_mas_max = lg_pixel_size_mas_max
        self.n_along = n_along
        self.n_across = n_across

        # resolutions = np.logspace(lg_pixel_size_mas_min, lg_pixel_size_mas_max, n_along)
        resolutions = np.sqrt(np.linspace(1, n_along, n_along))*10**lg_pixel_size_mas_min

        # 2D array of u.angle pixsizes
        self.pixsize_array = np.tile(resolutions, n_across).reshape(n_across, n_along).T*u.mas
        self.ft_class = ft_class
        self.ft_instance = None
        self.calculate_grid()
        self._image = None
        self._image_tau = None
        self.stokes = stokes

        # For compatibility with FT realization and possible future use
        # shift in mas
        self.dx = dx
        self.dy = dy
        # rotation angle in rad
        self.rot = rot

    def plot_resolutions(self):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 1)
        axes.plot(np.cumsum(self.pixsize_array[:, 0]),
                  np.cumsum(self.pixsize_array[:, :int(self.n_across/2)], axis=1)[:, -1])
        axes.set_xlabel("Along distance, mas")
        axes.set_ylabel("Across distance, mas")
        axes.set_aspect("auto")
        return fig

    def halfphi_app_max(self):
        """
        Maximum half opening-angle [rad] that can be imaged with given resolution.
        """
        return np.arctan(np.sum(self.pixsize_array[-1, :int(self.n_across/2)]) / np.sum(self.pixsize_array[:, 0]))

    def calculate_grid(self):
        """
        Calculate grid of ``(r_ob, d)`` - each point is in the center of the
        corresponding pixel. Thus, ``JetModelZoom.ft`` method should not shift
        phases on a half of a pixel. ``r_ob`` and ``d`` are in parsecs.
        """
        pc_x = np.cumsum(self.pix_to_pc, axis=0)-self.pix_to_pc/2
        pc_y_up = self.pix_to_pc[:, :self.pix_to_pc.shape[1]//2][::-1]
        pc_y_low = self.pix_to_pc[:, self.pix_to_pc.shape[1]//2:]
        pc_y_up = (np.cumsum(pc_y_up, axis=1) - pc_y_up/2)[::-1]
        pc_y_low = np.cumsum(pc_y_low, axis=1) - pc_y_low/2
        pc_y = np.hstack((pc_y_up[:, ::-1], -pc_y_low))
        self.r_ob = pc_x
        if not self.jet_side:
            self.r_ob *= -1
        # FIXME: In analogy with older convention we need ``-`` here
        self.d = -pc_y

    @property
    def pc_to_mas(self):
        return (u.pc/self.ang_to_dist).to(u.mas)

    @property
    def r_ob_mas(self):
        return self.r_ob*self.pc_to_mas

    @property
    def d_mas(self):
        return self.d*self.pc_to_mas

    @property
    def imgsize(self):
        return np.max(np.cumsum(self.pixsize_array, axis=0)), \
               np.max(np.cumsum(self.pixsize_array, axis=1))

    @property
    def img_extent(self):
        s0, s1 = self.imgsize[0].value, self.imgsize[1].value
        if self.jet_side:
            return 0, s0, -s1/2, s1/2
        else:
            return -s0, 0, -s1/2, s1/2

    def calculate_images(self, los_angle_deg, R_1_pc, b_0, m_b, K_1, n, s, gamma_min, background_fraction, Gamma_0, Gamma_1, betac_phi,
                         spiral_width_frac, phase_0, lambda_0, amp_0, phase_1, lambda_1, amp_1, phase_2, lambda_2, amp_2,
                         t_obs_days):
        self._image, self._image_tau = run_on_analytic_params_kh(self.z, los_angle_deg, R_1_pc,
                                                                 b_0, m_b,
                                                                 K_1, n,
                                                                 s, gamma_min, background_fraction,
                                                                 Gamma_0, Gamma_1, betac_phi,
                                                                 self.n_along, self.n_across, self.lg_pixel_size_mas_min, self.lg_pixel_size_mas_max,
                                                                 spiral_width_frac,
                                                                 phase_0, lambda_0, amp_0,
                                                                 phase_1, lambda_1, amp_1,
                                                                 phase_2, lambda_2, amp_2,
                                                                 t_obs_days,
                                                                 self.jet_side)

        self._image = np.atleast_2d(self._image)
        self._image_tau = np.atleast_2d(self._image_tau)

    def save_image_to_difmap_format(self, difmap_format_file, scale=1.0):
        with open(difmap_format_file, "w") as fo:
            for idx, imval in np.ndenumerate(self._image):
                if imval == 0:
                    continue
                j, i = idx
                dec = -self.r_ob_mas[i, j].value
                ra = -self.d_mas[i, j].value
                # print("RA = {}, DEC = {}".format(ra, dec))
                fo.write("{} {} {}\n".format(imval*scale, np.hypot(ra, dec), np.rad2deg(np.arctan2(ra, dec))))

    def image(self):
        return self._image

    def image_intensity(self):
        # Factor that accounts non-uniform pixel size in plotting
        factor = (self.pixsize_array/np.min(self.pixsize_array))**2
        return self.image()/factor.T

    def image_tau(self):
        return self._image_tau

    def ft(self, uv, rot=None):
        if rot is None:
            rot = self.rot
        # print("Doing FT of jet model with rot = {}".format(np.rad2deg(rot)))
        mas_to_rad = u.mas.to(u.rad)
        rot = np.array([[np.cos(rot), -np.sin(rot)],
                        [np.sin(rot), np.cos(rot)]])

        # No need in half pixel shifts cause coordinates are already at pixel
        # centers
        shift = [self.dx*mas_to_rad, self.dy*mas_to_rad]
        result = np.exp(-2.0*np.pi*1j*(uv @ shift))
        uv = uv @ rot

        x = (self.d*u.pc/self.ang_to_dist).to(u.rad).value
        y = (self.r_ob*u.pc/self.ang_to_dist).to(u.rad).value
        ft_instance = self.ft_class(uv, x.ravel(), y.ravel())
        img = self.image()
        result *= ft_instance.forward(img.T.ravel())
        del ft_instance, x, y

        return result

    @property
    @functools.lru_cache()
    def ang_to_dist(self):
        return self.cosmo.kpc_proper_per_arcmin(self.z)

    @property
    @functools.lru_cache()
    def pix_to_pc(self):
        """
        2D array of pixel sizes in parsecs.
        """
        return (self.pixsize_array * self.ang_to_dist).to(u.pc).value


class TwinJetImage(object):

    def __init__(self, z, n_along, n_across, lg_pixel_size_mas_min, lg_pixel_size_mas_max, ft_class=FINUFFT_NUNU,
                 dx=0.0, dy=0.0, rot=0.0, stokes="I"):
        self.jet = JetImage(z, n_along, n_across, lg_pixel_size_mas_min, lg_pixel_size_mas_max, ft_class,
                            True, dx, dy, rot, stokes)
        self.counterjet = JetImage(z, n_along, n_across, lg_pixel_size_mas_min, lg_pixel_size_mas_max, ft_class,
                                   False, dx, dy, rot, stokes)
        self.stokes = stokes
        self.rot = rot

    def set_rot(self, rot):
        self.rot = rot
        self.jet.rot = rot
        self.counterjet.rot = rot

    def calculate_images(self, los_angle_deg, R_1_pc, b_0, m_b, K_1, n, s, gamma_min, background_fraction, Gamma_0, Gamma_1, betac_phi,
                         spiral_width_frac, phase_0, lambda_0, amp_0, phase_1, lambda_1, amp_1, phase_2, lambda_2, amp_2,
                         t_obs_days):
        self.jet.calculate_images(los_angle_deg, R_1_pc, b_0, m_b, K_1, n, s, gamma_min, background_fraction, Gamma_0, Gamma_1, betac_phi,
                                  spiral_width_frac, phase_0, lambda_0, amp_0, phase_1, lambda_1, amp_1, phase_2, lambda_2, amp_2, t_obs_days)
        self.counterjet.calculate_images(los_angle_deg, R_1_pc, b_0, m_b, K_1, n, s, gamma_min, background_fraction, Gamma_0, Gamma_1, betac_phi,
                                         spiral_width_frac, phase_0, lambda_0, amp_0, phase_1, lambda_1, amp_1, phase_2, lambda_2, amp_2, t_obs_days)

    def ft(self, uv, rot=None):
        if rot is None:
            rot = self.rot
        return self.jet.ft(uv, rot) + self.counterjet.ft(uv, rot)

    def substitute_uvfits(self, in_uvfits, out_uvfits, average_freq=False, use_V=False, noise_scale_factor=1.0,
                          need_downscale_uv=None):
        uvdata = UVData(in_uvfits)
        noise = uvdata.noise(average_freq=average_freq, use_V=use_V)
        for baseline, baseline_noise_std in noise.items():
            noise.update({baseline: noise_scale_factor*baseline_noise_std})
        uvdata.zero_data()
        uvdata.substitute([self])
        uvdata.noise_add(noise)
        if need_downscale_uv is None:
            need_downscale_uv = downscale_uvdata_by_freq(uvdata)
        uvdata.save(out_uvfits, rewrite=True, downscale_by_freq=need_downscale_uv)

    def save_image_to_difmap_format(self, difmap_format_file, scale=1.0):
        image = np.hstack((self.counterjet._image[::-1], self.jet._image))
        # CJ has its DEC coordinates multiplied on -1.
        r_ob_mas = np.vstack((self.counterjet.r_ob_mas, self.jet.r_ob_mas))
        d_mas = np.vstack((self.counterjet.d_mas, self.jet.d_mas))

        with open(difmap_format_file, "w") as fo:
            for idx, imval in np.ndenumerate(image):
                if imval == 0:
                    continue
                j, i = idx
                dec = -r_ob_mas[i, j].value
                ra = -d_mas[i, j].value
                # print("RA = {}, DEC = {}".format(ra, dec))
                fo.write("{} {} {}\n".format(imval*scale, np.hypot(ra, dec), np.rad2deg(np.arctan2(ra, dec))))


class Fitter(object):
    def __init__(self, z, n_along, n_across, lg_pixel_size_mas_min, lg_pixel_size_mas_max, only_closures=True,
                 phase_penalty=False, dx_mas=0.0, dy_mas=0.0):
        self.khjet = TwinJetImage(z, n_along, n_across, lg_pixel_size_mas_min, lg_pixel_size_mas_max, dx=dx_mas, dy=dy_mas)
        self.uv_df = None
        self.camp_df = None
        self.cph_df = None
        # Predictions
        self.mu_amp = None
        self.mu_camp = None
        self.mu_cphase = None
        # Data file
        self.uvfits = None
        self.only_closures = only_closures
        self.phase_penalty = phase_penalty

    def make_data(self, uvfits, avg_time_logcamp=60, avg_time_cph=60, snr_cut_logcamp=0., snr_cut_cph=0.,
                  avg_time_amp=60, snr_cut_amp=4):
        self.uvfits = uvfits

        # Fitting only first IF
        # obs = eh.obsdata.load_uvfits(uvfits, IF=[0])
        obs = eh.obsdata.load_uvfits(self.uvfits)

        # factor = obs.estimate_noise_rescale_factor()
        # obs = obs.rescale_noise(factor)

        print(obs.tarr)
        # Sort array by median SNR!
        obs.reorder_tarr_snr()
        print(obs.tarr)

        if not self.only_closures:
            obs.add_amp(avg_time=avg_time_amp, snrcut=snr_cut_amp, return_type="df", debias=True)
            amp_df = obs.amp
            amp_uv = amp_df[["u", "v"]]

        # Adds attribute self.logcamp
        obs.add_logcamp(avg_time=avg_time_logcamp, return_type='df', ctype='logcamp',
                        count='min', debias=True, snrcut=snr_cut_logcamp,
                        num_samples=1000, round_s=0.1)
        camp_df = obs.logcamp[["time", "t1", "t2", "t3", "t4", "u1", "v1", "u2", "v2", "u3", "v3", "u4", "v4", "camp", "sigmaca"]]

        camp_uv = pd.DataFrame(np.vstack((camp_df[["u1", "v1"]].values,
                                          camp_df[["u2", "v2"]].values,
                                          camp_df[["u3", "v3"]].values,
                                          camp_df[["u4", "v4"]].values)), columns=["u", "v"])
        # Adds attribute self.cphase
        obs.add_cphase(avg_time=avg_time_cph, return_type='df', count='min', snrcut=snr_cut_cph,
                       num_samples=1000, round_s=0.1, uv_min=False)
        cph_df = obs.cphase[["time", "t1", "t2", "t3", "u1", "v1", "u2", "v2", "u3", "v3", "cphase", "sigmacp"]]
        cph_df["cphase"] = np.deg2rad(cph_df["cphase"])
        cph_df["sigmacp"] = np.deg2rad(cph_df["sigmacp"])
        cph_uv = pd.DataFrame(np.vstack((cph_df[["u1", "v1"]].values,
                                         cph_df[["u2", "v2"]].values,
                                         cph_df[["u3", "v3"]].values)), columns=["u", "v"])

        if self.only_closures:
            uv_df = pd.concat([camp_uv, cph_uv])
        else:
            uv_df = pd.concat([camp_uv, cph_uv, amp_uv])
        uv_df = uv_df.drop_duplicates().reset_index(drop=True)

        # For each camp & cphase find vectors (camp_indx1, .., camp_indx4) and (cphase_indx1, ..., cphase_indx3) with indices
        # of (u1, v1), (u2, v2) in common uv vector.
        camp_indx1 = list()
        camp_indx2 = list()
        camp_indx3 = list()
        camp_indx4 = list()
        cphase_indx1 = list()
        cphase_indx2 = list()
        cphase_indx3 = list()


        if not self.only_closures:
            amp_indx = list()
            for _, row in amp_df.iterrows():
                amp_indx.append(uv_df.index[np.logical_and(uv_df['u'] == row["u"], uv_df["v"] == row["v"])][0])

        for _, row in camp_df.iterrows():
            camp_indx1.append(uv_df.index[np.logical_and(uv_df['u'] == row["u1"], uv_df["v"] == row["v1"])][0])
            camp_indx2.append(uv_df.index[np.logical_and(uv_df['u'] == row["u2"], uv_df["v"] == row["v2"])][0])
            camp_indx3.append(uv_df.index[np.logical_and(uv_df['u'] == row["u3"], uv_df["v"] == row["v3"])][0])
            camp_indx4.append(uv_df.index[np.logical_and(uv_df['u'] == row["u4"], uv_df["v"] == row["v4"])][0])
        for _, row in cph_df.iterrows():
            cphase_indx1.append(uv_df.index[np.logical_and(uv_df['u'] == row["u1"], uv_df["v"] == row["v1"])][0])
            cphase_indx2.append(uv_df.index[np.logical_and(uv_df['u'] == row["u2"], uv_df["v"] == row["v2"])][0])
            cphase_indx3.append(uv_df.index[np.logical_and(uv_df['u'] == row["u3"], uv_df["v"] == row["v3"])][0])

        camp_df["indx1"] = camp_indx1
        camp_df["indx2"] = camp_indx2
        camp_df["indx3"] = camp_indx3
        camp_df["indx4"] = camp_indx4
        cph_df["indx1"] = cphase_indx1
        cph_df["indx2"] = cphase_indx2
        cph_df["indx3"] = cphase_indx3
        self.camp_df = camp_df[["camp", "sigmaca", "indx1", "indx2", "indx3", "indx4"]]
        self.cph_df = cph_df[["cphase", "sigmacp", "indx1", "indx2", "indx3"]]
        self.uv_df = uv_df
        if not self.only_closures:
            amp_df["indx"] = amp_indx
            self.amp_df = amp_df[["amp", "sigma", "indx"]]

    def calculate_prediction(self, rot, los_angle_deg, R_1_pc, b_0, m_b, K_1, n, s, gamma_min, background_fraction, Gamma_0, Gamma_1, betac_phi,
                             spiral_width_frac, phase_0, lambda_0, amp_0, phase_1, lambda_1, amp_1, phase_2, lambda_2, amp_2, t_obs_days):
        self.khjet.calculate_images(los_angle_deg, R_1_pc, b_0, m_b, K_1, n, s, gamma_min, background_fraction, Gamma_0, Gamma_1, betac_phi,
                                    spiral_width_frac, phase_0, lambda_0, amp_0, phase_1, lambda_1, amp_1, phase_2, lambda_2, amp_2, t_obs_days)
        self.khjet.rot = rot
        # FIXME: Get 2D uv array from self.uv_df dataframe
        mu_vis = self.khjet.ft(self.uv_df[["u", "v"]].values, rot)
        mu_real = mu_vis.real
        mu_imag = mu_vis.imag
        mu_amp = np.hypot(mu_real, mu_imag)
        mu_phase = np.arctan2(mu_imag, mu_real)

        if not self.only_closures:
            # Calculating amplitudes
            self.mu_amp = mu_amp[self.amp_df["indx"].values]

        # Calculating close amplitude
        # 1-2
        # FIXME: Get indexes from dataframes
        self.mu_camp = mu_amp[self.camp_df["indx1"].values]
        # 3-4
        self.mu_camp *= mu_amp[self.camp_df["indx2"].values]
        # 1-4
        self.mu_camp /= mu_amp[self.camp_df["indx3"].values]
        # 2-3
        self.mu_camp /= mu_amp[self.camp_df["indx4"].values]

        # Working with log of closure amplitude
        self.mu_camp = np.log(self.mu_camp)

        # Calculating close phase
        # 1-2
        self.mu_cphase = mu_phase[self.cph_df["indx1"].values]
        # 2-3
        self.mu_cphase += mu_phase[self.cph_df["indx2"].values]
        # 3-1
        self.mu_cphase += mu_phase[self.cph_df["indx3"].values]

    def logl(self):
        norm_pdf_logC = np.log(np.sqrt(2*np.pi))

        r_amp = (self.camp_df["camp"].values - self.mu_camp)/self.camp_df["sigmaca"].values
        loglik_camp = -np.log(self.camp_df["sigmaca"].values) - norm_pdf_logC - 0.5*r_amp*r_amp

        r_cph = (self.cph_df["cphase"].values - self.mu_cphase)/self.cph_df["sigmacp"].values
        loglik_cph = -np.log(self.cph_df["sigmacp"].values) - norm_pdf_logC - 0.5*r_cph*r_cph

        return loglik_camp.sum() + loglik_cph.sum()

    def chisq(self):
        if not self.only_closures:
            chisq_amp = np.sum( (self.amp_df["amp"].values - self.mu_amp)**2/self.amp_df["sigma"]**2 )/len(self.mu_amp)
        else:
            chisq_amp = 0.0
        chisq_camp = np.sum(((self.camp_df["camp"].values - self.mu_camp)/self.camp_df["sigmaca"].values)**2)/len(self.camp_df["camp"])
        chisq_cph = 2.0*np.sum((1.0 - np.cos(self.cph_df["cphase"].values - self.mu_cphase))/self.cph_df["sigmacp"].values**2)/len(self.cph_df["cphase"])
        print("========================== Chisq = ", chisq_amp + chisq_camp + chisq_cph)
        return chisq_amp + chisq_cph + chisq_camp

    def chisq_params(self, rot_deg, R_1_pc, lgb_0, m_b, lgbkg, Gamma_0, Gamma_1, beta_phi, spiral_width_frac,
                     phase_0, lambda_0, amp_0,
                     phase_1, lambda_1, amp_1,
                     phase_2, lambda_2, amp_2,
                     t_obs_days):
        print("rot = {:.1f} deg, R_1_pc = {:.2f}, m_b = {:.3f},, m_b = {:.2f}, bkg = {:.4f}, Gamma_0 = {:.2f},"
              " Gamma_1 = {:.2f}, beta_phi = {:.3f}, width_frac = {:.3f}, phase_0 = {:.2f}, lambda_0 = {:.2f}, amp_0 = {:.3f},"
              " phase_1 = {:.2f}, lambda_1 = {:.2f}, amp_1 = {:.3f},  phase_2 = {:.2f}, lambda_2 = {:.2f}, amp_2 = {:.3f}".format(rot_deg, R_1_pc, 10**lgb_0, m_b, 10**lgbkg,
                                                                            Gamma_0, Gamma_1, beta_phi,
                                                                            spiral_width_frac,
                                                                            phase_0, lambda_0, amp_0,
                                                                            phase_1, lambda_1, amp_1,
                                                                            phase_2, lambda_2, amp_2))
        self.calculate_prediction(rot=np.deg2rad(rot_deg), los_angle_deg=17., R_1_pc=R_1_pc, b_0=10**lgb_0, m_b=m_b, K_1=1.5,
                                  n=1.0, s=2.1, gamma_min=10, background_fraction=10**lgbkg, Gamma_0=Gamma_0, Gamma_1=Gamma_1, betac_phi=beta_phi,
                                  spiral_width_frac=spiral_width_frac,
                                  amp_0=amp_0, phase_0=phase_0, lambda_0=lambda_0,
                                  amp_1=amp_1, phase_1=phase_1, lambda_1=lambda_1,
                                  amp_2=amp_2, phase_2=phase_2, lambda_2=lambda_2,
                                  t_obs_days=t_obs_days)
        result = self.chisq()
        # if self.phase_penalty:
        #     result += 1/np.sin(phase_1-phase_0)**4
        return result


def dlib_func(fitter):
    def func(rot_deg, R_1_pc, m_b, lgbkg, Gamma_0, Gamma_1, beta_phi, spiral_width_frac,
             phase_0, lambda_0, amp_0,
             phase_1, lambda_1, amp_1,
             phase_2, lambda_2, amp_2):
        return fitter.chisq_params(rot_deg, R_1_pc, m_b, lgbkg, Gamma_0, Gamma_1, beta_phi, spiral_width_frac,
                                   phase_0, lambda_0, amp_0,
                                   phase_1, lambda_1, amp_1,
                                   phase_2, lambda_2, amp_2)
    return func


def bobyqa_func(fitter):
    def func(params):
        params_list = params.tolist()
        return fitter.chisq_params(*params_list)
    return func


if __name__ == "__main__":
    # template_uvfits = "/home/ilya/data/alpha/RA/1228+126.u.2014_03_26.uvf"
    # template_uvfits = "/home/ilya/data/alpha/RA/MOJAVE_pics/1228+126.u.2000_01_22.uvf"
    # template_uvfits = "/home/ilya/data/alpha/BK145/1228+126.X.2009_05_23_ta60.uvf_cal"
    template_uvfits = "/home/ilya/data/alpha/BK145/1228+126.U.2009_05_23C_ta60.uvf_cal"
    # uvdata = UVData(template_uvfits)
    # fig = uvdata.uvplot()
    # plt.show()

    # art_uvfits = "/home/ilya/data/alpha/RA/1228+126.u.2014_03_26_art_kh.uvf"
    # art_uvfits = "/home/ilya/data/alpha/RA/BK145_8GHz_artificial_KH.uvf"
    art_uvfits = "/home/ilya/data/alpha/RA/BK145_15GHz_artificial_KH.uvf"
    art_ccfits = "/home/ilya/data/alpha/RA/BK145_art_kh_cc_15GHz.fits"
    # real_ccfits = "/home/ilya/data/alpha/RA/BK145_Lesha.fits"
    path_to_script = "/home/ilya/github/bk_transfer/scripts/final_clean_nw"
    mapsize = (1024, 0.2)
    # mapsize = (1024, 0.1)
    # To quickly show J + CJ images for freq_ghz GHz:
    # j = np.loadtxt("jet_image_i_{}.txt".format(freq_ghz)); cj = np.loadtxt("cjet_image_i_{}.txt".format(freq_ghz)); jcj = np.hstack((cj[::, ::-1], j)); plt.matshow(jcj, aspect="auto");plt.colorbar(); plt.show()

    only_closures = True
    # fitter = Fitter(0.00436, 400, 80, -2, -0.5, only_closures=only_closures)
    # ''dx>0`` shifts to negative RA.
    # ''dy>0`` shifts to negative DEC.
    fitter = Fitter(0.00436, 1000, 200, -2.5, -0.5, only_closures=only_closures,
                    dx_mas=-0.5*np.cos(np.deg2rad(20)), dy_mas=0.5*np.sin(np.deg2rad(20)))
    fitter.make_data(template_uvfits, snr_cut_logcamp=4)


    # fitter.calculate_prediction(rot=np.deg2rad(-108), los_angle_deg=17., R_1_pc=0.14, b_0=0.085, m_b=0.64, K_1=1.5,
    #                             n=1.0, s=2.1, gamma_min=10, background_fraction=0.01, Gamma_0=1.8, Gamma_1=0.5, betac_phi=0.0,
    #                             phase_0=5.5, lambda_0=39.4, amp_0=0.9)
    # jimage = fitter.khjet.jet.image()
    # cjimage = fitter.khjet.counterjet.image()
    # import matplotlib.pyplot as plt
    # plt.matshow(np.log(jimage), cmap="plasma")
    # plt.matshow(np.log(cjimage), cmap="plasma")
    # plt.show()
    # chisq = fitter.chisq()
    # print(chisq)
    # sys.exit(0)


    # rot, R_1_pc, lgb_0, m_b, lgbkg, Gamma_0, Gamma_1, beta_phi, spiral_width_frac, phase_0, lambda_0
    lower_bounds = [-120, 0.07, -3, 0.3, -3.0, 1.01, 0.01, -0.15, 0.001,      0, 20]
    upper_bounds = [-100, 0.13, -1, 1.0, -2.3, 3.01, 1.00,  0.15,  0.05,  np.pi, 50]
    to_minimize = dlib_func(fitter)

    n_fun_eval = 1000
    # result = dlib.find_min_global(to_minimize, lower_bounds, upper_bounds, n_fun_eval, 0.5)
    # print(result)
    # best_params = result[0]


    # x0 = [-103.62814028137609, 0.0705690157130614, 0.7061685115385884, -2.3104247966120823, 2.0013630403498084,
    #       0.33429269102313619, 0.014215505797217826, 3.1940716279895605, 45.57515672309895]
    # to_minimize = bobyqa_func(fitter)
    # x0 = np.array([-106.455133955111, 0.08779788366143099, 0.8067411419671657, -2.989604541723953, 1.094261422796554,
    #       0.9225556631956895, 0.13445043733897255, 4.4215900603146485, 29.578739217651314])
    # soln = pybobyqa.solve(to_minimize, x0, maxfun=n_fun_eval, bounds=(lower_bounds, upper_bounds),
    #                       seek_global_minimum=True, print_progress=True, scaling_within_bounds=True,
    #                       rhoend=1e-4)
    # best_params = soln.x
    # print(soln.f)
    # print(best_params)

    # Best BK73 ########################################################################################################
    # TODO: Should I fix Gamma_0 = 1
    # W/o B_1 - better
    # best_params = [-100.32550816308886, 0.11697396773578071, 0.7701131403737128, -2.3188592721462196,
    #                1.1484269375718186, 0.7067622985150728, 0.11002834191905761, 1.0644699179418908, 28.851699600545345]
    # With B_1
    # best_params = [-110.52044308658354, 0.0722456186311913, -1.1499009645827187, 0.6994892555042932, -2.362447214542485,
    #                2.8552034525654864, 0.8471196869611207, -0.09667613023303227, 1.4096000711664451, 43.22118075590454]
    # Bobyqa 6000, w B_1, closures: chisq 21
    best_params = [-109.1, 0.10, np.log10(0.103), 0.72, np.log10(0.0050), 1.04, 0.21, -0.145, 1.44, 28.91]
    # dlib 6000, w B_1, closures: chisq 34
    # ([-110.1001709648304, 0.08533483762099554, -0.9714827080221451, 0.6893056926911909, -2.976577552113481,
    #   1.1427513728424807, 0.06948140460911877, 0.13482606220389967, 1.9058339348629993, 49.637004116433246], 34.78735990998545)


    # Phases like in Nikonov Crimea slides, chisq = 122
    # best_params = [-107.6, 0.102, 0.780, np.log10(0.0021), 1.01, 0.95, 0.106, 1.88, 27.67] #, 5.04, 26.73]
    # Currently the best
    # best_params = [-109.61811599357868, 0.08629745825684192, 0.7622885034131177, -2.4369768002613994,
    #                1.0108506486764424, 0.8785702110787783, -0.11470488987515165, 3.253190234858707, 49.13642096099992,
    #                6.235115977915566, 33.97762904033963]


    # fitter.make_data(template_uvfits, snr_cut_logcamp=4)
    # if only_closures:
    #     best_b_0 = best_params[2]
    #     i = 1
    # else:
    #     best_b_0 = best_params[2]
    #     i = 1




    # Making BK145
    # -110.0, 0.137, np.log10(0.075), 0.65, np.log10(0.005), 1.25, 0.10, 0.0, 0.05 - 1
    # -110.0, 0.137, np.log10(0.065), 0.65, np.log10(0.005), 1.01, 0.75, 0.0, 0.05 - 2
    # -110.0, 0.13, np.log10(0.19), 0.5, np.log10(0.001), 1.1, 0.50, 0.0, 0.025 - 3
    # best_params = [-110.0, 0.13, np.log10(0.125), 0.5, np.log10(0.005), 1.1, 0.50, 0.0, 0.05,
    #                # R = 0.09367584
    #                 np.deg2rad(80.0), 8.02, 0.72,
    #                # R = 0.12311555
    #                 np.deg2rad(212.0), 7.61, 0.95,
    #                # R = 0.05380604
    #                 np.deg2rad(66), 2.57, 0.41]

    best_params = [-110.0, 0.13, np.log10(0.16), 0.5, np.log10(0.001), 1.1, 0.50, 0.0, 0.025,
                   # R = 0.09367584
                   np.deg2rad(80.0), 8.02, 0.72,
                   # R = 0.12311555
                   np.deg2rad(212.0), 7.61, 0.95,
                   # R = 0.05380604
                   np.deg2rad(66), 2.57, 0.41]

    # np.deg2rad(80.0), 8.02, 0.68,
# np.deg2rad(212.0), 7.61, 0.9,
# np.deg2rad(66), 2.57, 0.39]
    t_obs_days = 0.0
    for t_obs_days in np.linspace(0, int(365*50*20), int(20)):
        print("t = ", t_obs_days)
        fitter.calculate_prediction(rot=np.deg2rad(best_params[0]), los_angle_deg=17., R_1_pc=best_params[1], b_0=10**best_params[2],
                                    m_b=best_params[3], K_1=1.5, n=1.0, s=2.0, gamma_min=10,
                                    background_fraction=10**best_params[4],
                                    Gamma_0=best_params[5], Gamma_1=best_params[6], betac_phi=best_params[7],
                                    spiral_width_frac=best_params[8],
                                    phase_0=best_params[9], lambda_0=best_params[10], amp_0=best_params[11],
                                    phase_1=best_params[12], lambda_1=best_params[13], amp_1=best_params[14],
                                    phase_2=best_params[15], lambda_2=best_params[16], amp_2=best_params[17],
                                    t_obs_days=t_obs_days)

        # fitter.khjet.calculate_images(los_angle_deg=17., R_1_pc=best_params[1], b_0=10**best_params[2],
        #                               m_b=best_params[3], K_1=1.5, n=1.0, s=2.1, gamma_min=10,
        #                               background_fraction=10**best_params[4],
        #                               Gamma_0=best_params[5], Gamma_1=best_params[6], betac_phi=best_params[7],
        #                               spiral_width_frac=best_params[8],
        #                               phase_0=best_params[9], lambda_0=best_params[10], amp_0=best_params[11],
        #                               phase_1=best_params[12], lambda_1=best_params[13], amp_1=best_params[14],
        #                               phase_2=best_params[15], lambda_2=best_params[16], amp_2=best_params[17],
        #                               t_obs_days=t_obs_days)

        jimage = fitter.khjet.jet.image()
        cjimage = fitter.khjet.counterjet.image()
        import matplotlib.pyplot as plt
        # plt.matshow(np.log(jimage), cmap="plasma")
        # plt.matshow(np.log(cjimage), cmap="plasma")
        # plt.show()
        jcj = np.hstack((cjimage[::-1, ::-1], jimage[::-1, ::]))
        plt.matshow(np.log(jcj), aspect="auto", cmap="inferno")
        # plt.colorbar()
        plt.savefig("jcj_lttd_t_{:.1f}_yrs.png".format(t_obs_days/365), bbox="tight")
        break
    # plt.show()

    print("Flux = ", np.sum(jimage) + np.sum(cjimage))

    # sys.exit(0)

#
# # chisq = fitter.chisq()
#     # print(chisq)
#     # Make artificial data
    fitter.khjet.substitute_uvfits(template_uvfits, art_uvfits)
    uvdata = UVData(art_uvfits)
#     # Plot radplot
    fig = uvdata.uvplot()
    fig.savefig("/home/ilya/data/alpha/RA/kh_radplot_15GHz.png", bbox_inches="tight", dpi=300)
    plt.show()
    sys.exit(0)

    #
#     import matplotlib.pyplot as plt
#     plt.close()
    # CLEAN artificial data
    clean_difmap(fname=art_uvfits,
                 outfname=art_ccfits, stokes="i",
                 mapsize_clean=mapsize, path_to_script=path_to_script,
                 show_difmap_output=False)
    # Make CLEAN map
    ccimage = create_clean_image_from_fits_file(art_ccfits)
    real_ccimage = create_clean_image_from_fits_file(real_ccfits)
    beam = ccimage.beam
    npixels_beam = np.pi*beam[0]*beam[1]/(4*np.log(2)*mapsize[1]**2)
    std = find_image_std(ccimage.image, beam_npixels=npixels_beam)
    blc, trc = find_bbox(ccimage.image, level=4*std, min_maxintensity_mjyperbeam=30*std, min_area_pix=30*npixels_beam,
                         delta=10)
    print(blc, trc)
    # blc = (450, 450)
    # trc = (980, 750)

    blc = (408, 460)
    trc = (889, 693)

    fig = iplot(ccimage.image, x=ccimage.x, y=ccimage.y,
                # min_abs_level=0.001*0.216,
                min_abs_level=3*std,
                blc=blc, trc=trc, beam=beam,
                close=True, show_beam=True, show=False, contour_color='gray', contour_linewidth=0.25, beam_place="lr")
    fig.savefig("/home/ilya/data/alpha/RA/art_kh_8GHz.png", dpi=600, bbox_inches="tight")

    fig = iplot(ccimage.image, (real_ccimage.image - ccimage.image)/real_ccimage.image, x=ccimage.x, y=ccimage.y,
                # min_abs_level=0.001*0.216,
                min_abs_level=3*std,
                blc=blc, trc=trc, beam=beam,
                # colors_mask=ccimage.image < 0.001*0.216,
                colors_mask=ccimage.image < 3*std,
                cmap="bwr",
                close=True, show_beam=True, show=False, contour_color='gray', contour_linewidth=0.25, beam_place="lr",
                color_clim=[-1, 1])
    fig.savefig("/home/ilya/data/alpha/RA/obs_minus_art_frac_kh_8GHz.png", dpi=600, bbox_inches="tight")

    # plt.show()

    # sys.exit(0)




    # jimage = fitter.khjet.jet.image()
    # cjimage = fitter.khjet.counterjet.image()
    # fitter.khjet.save_image_to_difmap_format("/home/ilya/data/alpha/RA/art_kh.dfm")
    # rotate_difmap_model("/home/ilya/data/alpha/RA/art_kh.dfm",
    #                     "/home/ilya/data/alpha/RA/art_kh_rotated.dfm",
    #                     90.0)
    # convert_difmap_model_file_to_CCFITS("/home/ilya/data/alpha/RA/art_kh_rotated.dfm", "I", mapsize,
    #                                     (0.25, 0.25, 0), template_uvfits,
    #                                     "/home/ilya/data/alpha/RA/art_kh_rotated_convolved_cc.fits")
    # ccimage = create_clean_image_from_fits_file("/home/ilya/data/alpha/RA/art_kh_rotated_convolved_cc.fits")
    # blc = (480, 480)
    # trc = (800, 544)
    # fig = iplot(ccimage.image, np.log(ccimage.image), colors_mask=ccimage.image < 0.00001*np.max(ccimage.image), cmap="nipy_spectral",
    #             x=ccimage.x, y=ccimage.y, min_abs_level=0.00001*np.max(ccimage.image), blc=blc, trc=trc, beam=(0.25, 0.25, 0),
    #             close=True, show_beam=True, show=False, contour_color='gray', contour_linewidth=0.25, beam_place="lr")
    # fig.savefig("/home/ilya/data/alpha/RA/art_kh_model_nolttd.png", dpi=600, bbox_inches="tight")

