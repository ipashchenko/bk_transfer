#include <string>
#include <iostream>
#include <fstream>
#include "ImagePlane.h"
#include "Observation.h"
#include "BField.h"
#include "VField.h"
#include "Cone.h"
#include "Cylinder.h"
#include "Parabaloid.h"
#include "Jet.h"
#include "utils.h"
#include <cmath>
#include "NField.h"
#include "Pixel.h"
#include <ctime>
#include <chrono>

using Eigen::Vector3d;
using Eigen::Matrix3Xd;
using std::vector;
using std::pair;
using namespace boost::numeric::odeint;
namespace ph = std::placeholders;

typedef std::chrono::high_resolution_clock Clock;


std::vector<double> run_on_analytic() {
	auto t1 = Clock::now();
	std::clock_t start;
	start = std::clock();

	// FIXME: z = 0 case leads to NaN intersections
    // M87
//    double redshift = 0.00436;
//    double los_angle = 17.0*M_PI/180.0;
    double redshift = 0.5;
    double los_angle = 3.0*M_PI/180.0;

    // Observed frequencies in GHz
    std::vector<double> nu_observed_ghz{2.1};
    std::vector<double> total_fluxes;
    // Frequencies in the BH frame in Hz
    std::vector<double> nu_bh;
    for(auto nu_obs_ghz : nu_observed_ghz) {
        nu_bh.push_back(nu_obs_ghz*1E+09*(1+redshift));
    }

    // Setting geometry ================================================================================================
    Vector3d origin = {0., 0., 0.};
    Vector3d direction = {0., 0., 1.};
    double big_scale = 1000*pc;
    double cone_half_angle = 1.5*M_PI/180.0;
    Cone geometry(origin, direction, cone_half_angle, big_scale);
//    double R_cyl = 0.125*pc;
//    Cylinder geometry(origin, direction, R_cyl);
//    double R_1pc = 0.1*pc;
//    Parabaloid geometry(origin, direction, R_1pc, big_scale, true);

    // Setting B-field
    BKScalarBField bk_bfield(1.0, 1.0, &geometry);
//    HelicalConicalBField jetbfield(0.05, 0.5, 10.*M_PI/180., true, 0.0, &geometry);

    std::vector<VectorBField*> vbfields;
    std::vector<ScalarBField*> sbfields;
    sbfields.push_back(&bk_bfield);

    // Setting components of N-fields ==================================================================================
    double s = 2.5;
    double ds = 0.0;
    double gamma_min = 1.0;
    PowerLaw particles(s, gamma_min, "pairs");
    // Value at r=1pc
    double K_1 = 5000;
    // Exponent of the decrease
    double n = 2.0;
    BKNField bk_stat_nfield(K_1, n, &particles, true, &geometry);


    // Setting V-field =================================================================================================
    VField* vfield;
    bool central_vfield = true;
    double Gamma = 10.0;
    // Working with spirals
    //double Gamma = 1.20;
    if (central_vfield) {
        vfield = new ConstCentralVField(Gamma, &geometry, 0.0);
    } else {
        vfield = new ConstFlatVField(Gamma, &geometry, 0.05);
    }



    // Flare start time in sec
    // In months
    double t_start_1 = 30.0;
    t_start_1 *= 30.0 * 24.0 * 60.0 * 60.0;
    double t_start_2 = 10.0;
    t_start_2 *= 30.0 * 24.0 * 60.0 * 60.0;
    double flare_width_pc = 2.5;
    FlareBKNField bk_flare_nfield_1(10 * K_1, n, t_start_1, flare_width_pc, &particles, true, los_angle, redshift,
                                    &geometry, nullptr, vfield);
    FlareBKNField bk_flare_nfield_2(5 * K_1, n, t_start_2, flare_width_pc, &particles, true, los_angle, redshift,
                                    &geometry, nullptr, vfield);

    // Working spirals implementation
//    bk_stat_nfield.set_spiral(0.0, 30.0 * R_1pc, 0.9 * R_1pc);
//    bk_stat_nfield.set_spiral(M_PI, 30.0 * R_1pc, 0.9 * R_1pc);
//    bk_stat_nfield.set_spiral(M_PI / 6.0, 10.0 * R_1pc, 0.5 * R_1pc);

    std::vector<NField*> nfields;
    nfields.push_back(&bk_stat_nfield);
    nfields.push_back(&bk_flare_nfield_1);
    nfields.push_back(&bk_flare_nfield_2);


    Jet bkjet(&geometry, vfield, sbfields, vbfields, nfields);

    // FIXME: Put inside frequency loop for dep. on frequency
    // Setting parameters of pixels and image ==========================================================================
    int number_of_pixels_along = 1024;
    int number_of_pixels_across = 512;
    // Non-uniform pixel from ``pixel_size_mas_start`` (near BH) to ``pixel_size_mas_stop`` (image edges)
    double pixel_size_mas_start = pow(10.0, -2.0);
    double pixel_size_mas_stop = pow(10.0, -1.0);
    auto image_size = std::make_pair(number_of_pixels_across, number_of_pixels_along);
    auto pc_in_mas = mas_to_pc(redshift);
    std::cout << "pc_in_mas " << pc_in_mas << std::endl;
    // Log10 of pixel size in cm
    auto lg_pixel_size_start = log10(pixel_size_mas_start*pc_in_mas*pc);
    auto lg_pixel_size_stop = log10(pixel_size_mas_stop*pc_in_mas*pc);

    std::cout << "Setting pixel size (pc) from " << pow(10.0, lg_pixel_size_start)/pc << " to " << pow(10.0, lg_pixel_size_stop)/pc << std::endl;
    for(auto jet_side : {true, false}) {

        // Ignore CJ
        if(jet_side == false) {
            continue;
        }

        ImagePlane imagePlane(image_size, lg_pixel_size_start, lg_pixel_size_stop, los_angle, jet_side);
        // Array of pixel sizes in cm
        auto pixel_sizes = imagePlane.getPixelSizes();
        // Array of pixel solid angles in rad*rad
        std::vector<std::vector<double>> pixel_solid_angles;
        pixel_solid_angles.resize(pixel_sizes.size());

        for(unsigned long i=0; i < pixel_sizes.size(); i++) {
            pixel_solid_angles[i].resize(pixel_sizes[0].size());
            for(unsigned long j=0; j < pixel_sizes[0].size(); j++) {
                // Divide by ``pc_in_mas*pc`` to bring ``cm`` to ``mas`` at source redshift
                pixel_solid_angles[i][j] = (pixel_sizes[i][j]/(pc_in_mas*pc))*(pixel_sizes[i][j]/(pc_in_mas*pc))*mas_to_rad*mas_to_rad;
            }
        }

        // Array of scale factors. Divide resulting image on this to obtain flux density in Jy. Accounts for cosmological
        // scaling of intensity
        std::vector<std::vector<double>> scales;
        scales.resize(pixel_sizes.size());
        for(unsigned long i=0; i < pixel_sizes.size(); i++) {
            scales[i].resize(pixel_sizes[0].size());
            for(unsigned long j=0; j < pixel_sizes[0].size(); j++) {
                scales[i][j] = 1E-23*(1.+redshift)*(1.+redshift)*(1.+redshift)/pixel_solid_angles[i][j];
            }
        }

        Observation observation(&bkjet, &imagePlane);

        // FIXME: Put out of frequency loop - these do not depend on frequency
        // Setting transfer-specific parameters ========================================================================
        double tau_max = 30;
        double dt_max_pc = 0.01;
        double dt_max = pc*dt_max_pc;
        double tau_min_log10 = -20.0;
        double tau_min = pow(10.,tau_min_log10);
        int n_ = 100;
        double relerr = 1e-10;
//        double relerr = 1e-4;

        // Solve for all Stokes parameters ("full") or only full intensity ("I")?
        string polarization = "I";
//        string polarization = "full";

        for(int i_nu=0; i_nu < nu_observed_ghz.size(); i_nu++) {
            if(jet_side) {
                std::cout << "Running transfer for frequency " << nu_observed_ghz[i_nu] << " GHz for approaching jet" << std::endl;
            } else {
                std::cout << "Running transfer for frequency " << nu_observed_ghz[i_nu] << " GHz for counter-jet" << std::endl;
            }
            observation.observe(n_, tau_max, dt_max, tau_min, nu_bh[i_nu], polarization, relerr);
            string value = "tau";
            auto image_tau = observation.getImage(value);

            value = "I";
            double total_flux = 0.0;
            auto image_i = observation.getImage(value);
            for (unsigned long int i = 0; i < image_i.size(); ++i) {
                for (unsigned long int j = 0; j < image_i[i].size(); ++j) {
                    image_i[i][j] = image_i[i][j]/scales[i][j];
                    total_flux += image_i[i][j];
                }
            }

            if(jet_side == true){
                total_fluxes.push_back(total_flux);
            }

            value = "l";
            auto image_l = observation.getImage(value);

            std::fstream fs;
            // Remove trailing zeros: https://stackoverflow.com/a/46424921
            std::ostringstream oss;
            oss << std::setprecision(8) << std::noshowpoint << nu_observed_ghz[i_nu];
            std::string freq_name = oss.str();

            std::string file_tau, file_tau_fr, file_i, file_q, file_u, file_v, file_l;
            if(jet_side) {
                file_tau = "jet_image_tau_" + freq_name + ".txt";
                file_tau_fr = "jet_image_taufr_" + freq_name + ".txt";
                file_i = "jet_image_i_" + freq_name + ".txt";
                file_q = "jet_image_q_" + freq_name + ".txt";
                file_u = "jet_image_u_" + freq_name + ".txt";
                file_v = "jet_image_v_" + freq_name + ".txt";
                file_l = "jet_image_l_" + freq_name + ".txt";
            } else {
                file_tau = "cjet_image_tau_" + freq_name + ".txt";
                file_tau_fr = "cjet_image_taufr_" + freq_name + ".txt";
                file_i = "cjet_image_i_" + freq_name + ".txt";
                file_q = "cjet_image_q_" + freq_name + ".txt";
                file_u = "cjet_image_u_" + freq_name + ".txt";
                file_v = "cjet_image_v_" + freq_name + ".txt";
                file_l = "cjet_image_l_" + freq_name + ".txt";
            }

            // Remove old file
            std::remove(file_i.c_str());
            std::remove(file_q.c_str());
            std::remove(file_u.c_str());
            std::remove(file_v.c_str());
            std::remove(file_l.c_str());
            std::remove(file_tau.c_str());
            std::remove(file_tau_fr.c_str());

            fs.open(file_tau, std::ios::out | std::ios::app);
            if (fs.is_open()) {
                write_2dvector(fs, image_tau);
                fs.close();
            }

            fs.open(file_i, std::ios::out | std::ios::app);
            if (fs.is_open()) {
                write_2dvector(fs, image_i);
                fs.close();
            }

            fs.open(file_l, std::ios::out | std::ios::app);
            if (fs.is_open()) {
                write_2dvector(fs, image_l, pc);
                fs.close();
            }

            if (polarization == "full") {
                value = "Q";
                auto image_q = observation.getImage(value);
                for (unsigned long int i = 0; i < image_q.size(); ++i) {
                    for (unsigned long int j = 0; j < image_q[i].size(); ++j) {
                        image_q[i][j] = image_q[i][j]/scales[i][j];
                    }
                }

                value = "U";
                auto image_u = observation.getImage(value);
                for (unsigned long int i = 0; i < image_u.size(); ++i) {
                    for (unsigned long int j = 0; j < image_u[i].size(); ++j) {
                        image_u[i][j] = image_u[i][j]/scales[i][j];
                    }
                }

                value = "V";
                auto image_v = observation.getImage(value);
                for (unsigned long int i = 0; i < image_v.size(); ++i) {
                    for (unsigned long int j = 0; j < image_v[i].size(); ++j) {
                        image_v[i][j] = image_v[i][j]/scales[i][j];
                    }
                }

                value = "tau_fr";
                auto image_tau_fr = observation.getImage(value);

                fs.open(file_tau_fr, std::ios::out | std::ios::app);
                if (fs.is_open()) {
                    write_2dvector(fs, image_tau_fr);
                    fs.close();
                }

                fs.open(file_q, std::ios::out | std::ios::app);
                if (fs.is_open()) {
                    write_2dvector(fs, image_q);
                    fs.close();
                }

                fs.open(file_u, std::ios::out | std::ios::app);
                if (fs.is_open()) {
                    write_2dvector(fs, image_u);
                    fs.close();
                }

                fs.open(file_v, std::ios::out | std::ios::app);
                if (fs.is_open()) {
                    write_2dvector(fs, image_v);
                    fs.close();
                }
            }
        }
    }

	std::cout << "CPU Time: "
						<< (std::clock() - start) / (double) (CLOCKS_PER_SEC)
						<< " s" << std::endl;
	auto t2 = Clock::now();
	std::cout << "User time: "
						<< std::chrono::duration_cast<std::chrono::seconds>(
								t2 - t1).count()
						<< " s" << std::endl;

    return total_fluxes;
}


std::vector<double> run_on_analytic_params(double redshift, double los_angle_deg, double cone_half_angle_deg,
                                           double b_0, double m_b,
                                           double s, double gamma_min,
                                           double K_1, double n,
                                           double Gamma,
                                           int number_of_pixels_along, int number_of_pixels_across,
                                           double lg_pixel_size_mas_start, double lg_pixel_size_mas_stop,
                                           double t_obs,
                                           std::vector<double> flare_params) {
    auto t1 = Clock::now();
    std::clock_t start;
    start = std::clock();

    double los_angle = los_angle_deg*M_PI/180.0;

    // Observed frequencies in GHz
    std::vector<double> nu_observed_ghz{2.2, 8.4};
    std::vector<std::string> nu_observed_band{"S", "X"};
    std::vector<double> total_fluxes;
    // Frequencies in the BH frame in Hz
    std::vector<double> nu_bh;
    for(auto nu_obs_ghz : nu_observed_ghz) {
        nu_bh.push_back(nu_obs_ghz*1E+09*(1.0+redshift));
    }

    // Setting geometry ================================================================================================
    Vector3d origin = {0., 0., 0.};
    Vector3d direction = {0., 0., 1.};
    double big_scale = 1000*pc;
    double cone_half_angle = cone_half_angle_deg*M_PI/180.0;
    Cone geometry(origin, direction, cone_half_angle, big_scale);

    // Setting B-field
    BKScalarBField bk_bfield(b_0, m_b, &geometry);

    std::vector<VectorBField*> vbfields;
    std::vector<ScalarBField*> sbfields;
    sbfields.push_back(&bk_bfield);

    // Setting components of N-fields ==================================================================================
    PowerLaw particles(s, gamma_min, "pairs");
    BKNField bk_stat_nfield(K_1, n, &particles, true, &geometry);


    // Setting V-field =================================================================================================
    VField* vfield;
    bool central_vfield = true;
    if (central_vfield) {
        vfield = new ConstCentralVField(Gamma, &geometry, 0.0);
    } else {
        vfield = new ConstFlatVField(Gamma, &geometry, 0.05);
    }

    double frac_amp, t_start_month, flare_width_pc;
    int num_flares = flare_params.size()/3;
    std::vector<NField*> nfields;
    nfields.push_back(&bk_stat_nfield);

    FlareBKNField* bk_flare_nfield;
    for(int i = 0; i < num_flares; i++){
        frac_amp = flare_params[3*i + 0];
        t_start_month = flare_params[3*i + 1];
        // In sec
        t_start_month *= 30.0 * 24.0 * 60.0 * 60.0;
        flare_width_pc = flare_params[3*i + 2];
        bk_flare_nfield = new FlareBKNField(frac_amp*K_1, n, t_start_month, flare_width_pc, &particles,
                                            true, los_angle, redshift, &geometry, nullptr, vfield);
        nfields.push_back(bk_flare_nfield);
    }


    Jet bkjet(&geometry, vfield, sbfields, vbfields, nfields);

    // FIXME: Put inside frequency loop for dep. on frequency
    // Setting parameters of pixels and image ==========================================================================
    // Non-uniform pixel from ``pixel_size_mas_start`` (near BH) to ``pixel_size_mas_stop`` (image edges)
    double pixel_size_mas_start = pow(10.0, lg_pixel_size_mas_start);
    double pixel_size_mas_stop = pow(10.0, lg_pixel_size_mas_stop);
    auto image_size = std::make_pair(number_of_pixels_across, number_of_pixels_along);
    auto pc_in_mas = mas_to_pc(redshift);
    std::cout << "pc_in_mas " << pc_in_mas << std::endl;
    // Log10 of pixel size in cm
    auto lg_pixel_size_start = log10(pixel_size_mas_start*pc_in_mas*pc);
    auto lg_pixel_size_stop = log10(pixel_size_mas_stop*pc_in_mas*pc);

    std::cout << "Setting pixel size (pc) from " << pow(10.0, lg_pixel_size_start)/pc << " to " << pow(10.0, lg_pixel_size_stop)/pc << std::endl;
    for(auto jet_side : {true, false}) {

        // Ignore CJ
        if(jet_side == false) {
            continue;
        }

        ImagePlane imagePlane(image_size, lg_pixel_size_start, lg_pixel_size_stop, los_angle, jet_side);
        // Array of pixel sizes in cm
        auto pixel_sizes = imagePlane.getPixelSizes();
        // Array of pixel solid angles in rad*rad
        std::vector<std::vector<double>> pixel_solid_angles;
        pixel_solid_angles.resize(pixel_sizes.size());

        for(unsigned long i=0; i < pixel_sizes.size(); i++) {
            pixel_solid_angles[i].resize(pixel_sizes[0].size());
            for(unsigned long j=0; j < pixel_sizes[0].size(); j++) {
                // Divide by ``pc_in_mas*pc`` to bring ``cm`` to ``mas`` at source redshift
                pixel_solid_angles[i][j] = (pixel_sizes[i][j]/(pc_in_mas*pc))*(pixel_sizes[i][j]/(pc_in_mas*pc))*mas_to_rad*mas_to_rad;
            }
        }

        // Array of scale factors. Divide resulting image on this to obtain flux density in Jy. Accounts for cosmological
        // scaling of intensity
        std::vector<std::vector<double>> scales;
        scales.resize(pixel_sizes.size());
        for(unsigned long i=0; i < pixel_sizes.size(); i++) {
            scales[i].resize(pixel_sizes[0].size());
            for(unsigned long j=0; j < pixel_sizes[0].size(); j++) {
                scales[i][j] = 1E-23*(1.+redshift)*(1.+redshift)*(1.+redshift)/pixel_solid_angles[i][j];
            }
        }

        Observation observation(&bkjet, &imagePlane);

        // FIXME: Put out of frequency loop - these do not depend on frequency
        // Setting transfer-specific parameters ========================================================================
        double tau_max = 30;
        double dt_max_pc = 0.01;
        double dt_max = pc*dt_max_pc;
        double tau_min_log10 = -20.0;
        double tau_min = pow(10.,tau_min_log10);
        int n_ = 100;
        double relerr = 1e-10;

        // Solve for all Stokes parameters ("full") or only full intensity ("I")?
        string polarization = "I";
//        string polarization = "full";

        for(int i_nu=0; i_nu < nu_observed_ghz.size(); i_nu++) {
            if(jet_side) {
                std::cout << "Running transfer for frequency " << nu_observed_ghz[i_nu] << " GHz for approaching jet" << std::endl;
            } else {
                std::cout << "Running transfer for frequency " << nu_observed_ghz[i_nu] << " GHz for counter-jet" << std::endl;
            }
            observation.observe(n_, tau_max, dt_max, tau_min, nu_bh[i_nu], polarization, relerr, 30.0*24.0*60.0*60.0*t_obs);
            string value = "tau";
            auto image_tau = observation.getImage(value);

            value = "I";
            double total_flux = 0.0;
            auto image_i = observation.getImage(value);
            for (unsigned long int i = 0; i < image_i.size(); ++i) {
                for (unsigned long int j = 0; j < image_i[i].size(); ++j) {
                    image_i[i][j] = image_i[i][j]/scales[i][j];
                    total_flux += image_i[i][j];
                }
            }

            if(jet_side == true){
                total_fluxes.push_back(total_flux);
            }

            value = "l";
            auto image_l = observation.getImage(value);

            std::fstream fs;
            std::stringstream ss;
            std::string freq_name = nu_observed_band[i_nu];
            ss << std::fixed << std::setprecision(1) << std::showpoint << t_obs;
            std::string epoch_obs = ss.str();

            std::string file_tau, file_tau_fr, file_i, file_q, file_u, file_v, file_l;
            if(jet_side) {
                file_tau = "jet_image_tau_" + freq_name + "_" + epoch_obs + ".txt";
                file_tau_fr = "jet_image_taufr_" + freq_name + ".txt";
                file_i = "jet_image_i_" + freq_name + "_" + epoch_obs + ".txt";
                file_q = "jet_image_q_" + freq_name + ".txt";
                file_u = "jet_image_u_" + freq_name + ".txt";
                file_v = "jet_image_v_" + freq_name + ".txt";
                file_l = "jet_image_l_" + freq_name + ".txt";
            } else {
                file_tau = "cjet_image_tau_" + freq_name + ".txt";
                file_tau_fr = "cjet_image_taufr_" + freq_name + ".txt";
                file_i = "cjet_image_i_" + freq_name + ".txt";
                file_q = "cjet_image_q_" + freq_name + ".txt";
                file_u = "cjet_image_u_" + freq_name + ".txt";
                file_v = "cjet_image_v_" + freq_name + ".txt";
                file_l = "cjet_image_l_" + freq_name + ".txt";
            }

            // Remove old file
            std::remove(file_i.c_str());
            std::remove(file_q.c_str());
            std::remove(file_u.c_str());
            std::remove(file_v.c_str());
            std::remove(file_l.c_str());
            std::remove(file_tau.c_str());
            std::remove(file_tau_fr.c_str());

            fs.open(file_tau, std::ios::out | std::ios::app);
            if (fs.is_open()) {
                write_2dvector(fs, image_tau);
                fs.close();
            }

            fs.open(file_i, std::ios::out | std::ios::app);
            if (fs.is_open()) {
                write_2dvector(fs, image_i);
                fs.close();
            }

            fs.open(file_l, std::ios::out | std::ios::app);
            if (fs.is_open()) {
                write_2dvector(fs, image_l, pc);
                fs.close();
            }

            if (polarization == "full") {
                value = "Q";
                auto image_q = observation.getImage(value);
                for (unsigned long int i = 0; i < image_q.size(); ++i) {
                    for (unsigned long int j = 0; j < image_q[i].size(); ++j) {
                        image_q[i][j] = image_q[i][j]/scales[i][j];
                    }
                }

                value = "U";
                auto image_u = observation.getImage(value);
                for (unsigned long int i = 0; i < image_u.size(); ++i) {
                    for (unsigned long int j = 0; j < image_u[i].size(); ++j) {
                        image_u[i][j] = image_u[i][j]/scales[i][j];
                    }
                }

                value = "V";
                auto image_v = observation.getImage(value);
                for (unsigned long int i = 0; i < image_v.size(); ++i) {
                    for (unsigned long int j = 0; j < image_v[i].size(); ++j) {
                        image_v[i][j] = image_v[i][j]/scales[i][j];
                    }
                }

                value = "tau_fr";
                auto image_tau_fr = observation.getImage(value);

                fs.open(file_tau_fr, std::ios::out | std::ios::app);
                if (fs.is_open()) {
                    write_2dvector(fs, image_tau_fr);
                    fs.close();
                }

                fs.open(file_q, std::ios::out | std::ios::app);
                if (fs.is_open()) {
                    write_2dvector(fs, image_q);
                    fs.close();
                }

                fs.open(file_u, std::ios::out | std::ios::app);
                if (fs.is_open()) {
                    write_2dvector(fs, image_u);
                    fs.close();
                }

                fs.open(file_v, std::ios::out | std::ios::app);
                if (fs.is_open()) {
                    write_2dvector(fs, image_v);
                    fs.close();
                }
            }
        }
    }

    std::cout << "CPU Time: "
              << (std::clock() - start) / (double) (CLOCKS_PER_SEC)
              << " s" << std::endl;
    auto t2 = Clock::now();
    std::cout << "User time: "
              << std::chrono::duration_cast<std::chrono::seconds>(
                      t2 - t1).count()
              << " s" << std::endl;

    return total_fluxes;
}

int main2() {
    std::vector<double> total_fluxes;
    total_fluxes = run_on_analytic();
    for(auto total_flux: total_fluxes){
        std::cout << "Total flux [Jy] = " << total_flux << "\n";
    }
    return 0;
}





// To run in parallel when fil params_3ridges.txt has 3 parameter sets:
// parallel --files --results t_obs_{11} --joblog log --jobs 7 -a params_flaring_jet.txt -n 1 -m --colsep ' ' "./bk_transfer"
int main(int argc, char *argv[]) {

    std::vector<double> total_fluxes;
    std::vector<double> flare_params;

    int num_of_flares = (argc - 12) / 3;
    std::cout << "Number of flares : " << num_of_flares << "\n";
    int argc_pred = 12 + 3 * num_of_flares;

    if(argc != argc_pred){
        std::cout << argc << "\n";
        std::cout << "Supply redshift, LOS-angle (deg), Cone half-angle (deg),"
                     " B_1 [G], K_1 [cm^{-3}], Gamma, N_along, N_across,"
                     " lg_pixel_size_mas_start, lg_pixel_size_mas_stop,\n"
                     " t_obs (months), flare_params: [amp, t_start (month), width (pc)]\n" << "\n";
        return 1;
    }
    else {

        double redshift = atof(argv[1]);
        std::cout << "z = " << argv[1] << "\n";

        double los_angle_deg = atof(argv[2]);
        std::cout << "LOS (deg) = " << argv[2] << "\n";

        double cone_half_angle_deg = atof(argv[3]);
        std::cout << "Cone HA (deg) = " << argv[3] << "\n";

        double b_0 = atof(argv[4]);
        std::cout << "B_1 (G) = " << argv[4] << "\n";

        double K_1 = atof(argv[5]);
        std::cout << "K_1 = " << argv[5] << "\n";

        double Gamma = atof(argv[6]);
        std::cout << "Gamma = " << argv[6] << "\n";

        int number_of_pixels_along = atoi(argv[7]);
        std::cout << "N_along = " << argv[7] << "\n";

        int number_of_pixels_across = atoi(argv[8]);
        std::cout << "N_across = " << argv[8] << "\n";

        double lg_pixel_size_mas_start = atof(argv[9]);
        std::cout << "lg(min_pixel_size[mas]) = " << argv[9] << "\n";

        double lg_pixel_size_mas_stop = atof(argv[10]);
        std::cout << "lg(max_pixel_size[mas]) = " << argv[10] << "\n";

        double t_obs = atof(argv[11]);
        std::cout << "t_obs(months) = " << argv[11] << "\n";

        double frac_amp, t_start_month, flare_width_pc;
        for(int i = 0; i < num_of_flares; i++){
            frac_amp = atof(argv[11 + 3*i + 1]);
            std::cout << "Frac.amp = " << frac_amp << "\n";
            t_start_month = atof(argv[11 + 3*i + 2]);
            std::cout << "T_start (month) = " << t_start_month << "\n";
            flare_width_pc = atof(argv[11 + 3*i + 3]);
            std::cout << "Width (pc) = " << flare_width_pc << "\n";
            flare_params.push_back(frac_amp);
            flare_params.push_back(t_start_month);
            flare_params.push_back(flare_width_pc);
        }

        total_fluxes = run_on_analytic_params(redshift, los_angle_deg, cone_half_angle_deg,
                b_0, 1,
                2.0, 10.0,
                K_1, 2,
                Gamma,
                number_of_pixels_along, number_of_pixels_across,
                lg_pixel_size_mas_start, lg_pixel_size_mas_stop,
                t_obs, flare_params);
    }
    for(auto total_flux: total_fluxes){
        std::cout << "Total flux [Jy] = " << total_flux << "\n";
    }

    return 0;
}