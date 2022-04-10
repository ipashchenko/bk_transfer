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

#include "linspace.h"

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
    double redshift = 0.00436;
    double los_angle = 17.0*M_PI/180.0;
    // 3C 84
//    double redshift = 0.017559;
//    double los_angle = 20.0*M_PI/180.0;

    // Observed frequencies in GHz
    std::vector<double> nu_observed_ghz{8.1, 15.4};
//    std::vector<double> nu_observed_ghz{1.6, 4.8};
//    std::vector<double> nu_observed_ghz{24, 43};
//    std::vector<double> nu_observed_ghz{15.4};
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
//    double cone_half_angle = 1.5*M_PI/180.0;
//    Cone geometry(origin, direction, cone_half_angle, big_scale);
//    double R_cyl = 0.125*pc;
//    Cylinder geometry(origin, direction, R_cyl);
    double R_1pc = 0.117*pc;
    Parabaloid geometry(origin, direction, R_1pc, big_scale, true);

    // Setting B-field
    BKScalarBField bk_bfield(0.05, 0.75, &geometry);
//    HelicalConicalBField jetbfield(0.05, 0.5, 85.*M_PI/180., true, 0.0, &geometry);

    std::vector<VectorBField*> vbfields;
    std::vector<ScalarBField*> sbfields;
    sbfields.push_back(&bk_bfield);
//    vbfields.push_back(&jetbfield);


    // Setting components of N-fields ==================================================================================
    double s = 2.0;
    double ds = 0.01;
    double gamma_min = 10.0;
    PowerLaw particles(s, gamma_min, "pairs", false);

    // Single BK ///////////////////////////////////
    // Value at r=1pc
    double K_1 = 0.0425;
    // Exponent of the decrease
    double n = 1.5;
    BKNField bk_stat_nfield(K_1, n, &particles, false, &geometry);

//    // Triple ridges ///////////////////////////////
//    // Value at r=1pc
//    double K_1 = 0.5;
//    // Exponent of the decrease
//    double n = 1.5;
//    BKNField bk_stat_nfield(K_1, n, &particles, false, &geometry);
//    bk_stat_nfield.set_heating_profile(1.0, 0.9, 0.025, 5.0, 0.025, 0.01);


//    // Double ridges ///////////////////////////////
//    // Value at r=1pc
//    double K_1 = 0.5;
//    // Exponent of the decrease
//    double n = 1.5;
//    BKNField bk_stat_nfield(K_1, n, &particles, false, &geometry);
//    bk_stat_nfield.set_heating_profile(1.0, 0.9, 0.025, 0.0, 0.025, 0.01);

//    // Working spirals implementation /////////////
//    // Value at r=1pc
//    double K_1 = 3.0;
//    // Exponent of the decrease
//    double n = 1.5;
//    BKNField bk_stat_nfield(K_1, n, &particles, false, &geometry);
//    bk_stat_nfield.set_background_fraction(0.01);
//    bk_stat_nfield.set_spiral(0.0, 30.0 * R_1pc, 0.9 * R_1pc);
//    bk_stat_nfield.set_spiral(M_PI, 30.0 * R_1pc, 0.9 * R_1pc);
//    bk_stat_nfield.set_spiral(M_PI / 6.0, 10.0 * R_1pc, 0.5 * R_1pc);

    // Setting V-field =================================================================================================
    VField* vfield;
    bool central_vfield = false;
    double Gamma = 1.2;
    // Gamma(z) = Gamma_0 + Gamma_1*(z/1pc)^{0.5}
    // For bk & ridges
    double Gamma_0 = 1.1;
    double Gamma_1 = 1.0;
    // For KH
//    double Gamma_0 = 1.1;
//    double Gamma_1 = 1.5;
    // Working with spirals
    //double Gamma = 1.20;
//    if (central_vfield) {
//        vfield = new ConstCentralVField(Gamma, &geometry, 0.0);
//    } else {
//        vfield = new ConstFlatVField(Gamma, &geometry, 0.0);
//    }
//    vfield = new ConstParabolicVField(Gamma, &geometry, 0.0);
    vfield = new AccelParabolicVField(Gamma_0, Gamma_1, &geometry, 0.0);

    std::vector<NField*> nfields;
    nfields.push_back(&bk_stat_nfield);


    Jet bkjet(&geometry, vfield, sbfields, vbfields, nfields);

    // FIXME: Put inside frequency loop for dep. on frequency
    // Setting parameters of pixels and image ==========================================================================
    // Uniform pixel size
//    int number_of_pixels_along = 1400;
//    int number_of_pixels_across = 500;

    // FIXME: Check FT for imager_dev
    int number_of_pixels_along = 512;
    int number_of_pixels_across = 1024;
    double pixel_size_mas_start = pow(10.0, -1);
    double pixel_size_mas_stop = pow(10.0, -1);

    // Non-uniform pixel from ``pixel_size_mas_start`` (near BH) to ``pixel_size_mas_stop`` (image edges)
    // Uniform pixel size
//    double pixel_size_mas_start = pow(10.0, -1.5);
//    double pixel_size_mas_stop = pow(10.0, -1.5);
    auto image_size = std::make_pair(number_of_pixels_across, number_of_pixels_along);
    auto pc_in_mas = mas_to_pc(redshift);
    std::cout << "pc_in_mas " << pc_in_mas << std::endl;
    // Log10 of pixel size in cm
    auto lg_pixel_size_start = log10(pixel_size_mas_start*pc_in_mas*pc);
    auto lg_pixel_size_stop = log10(pixel_size_mas_stop*pc_in_mas*pc);

    std::cout << "Setting pixel size (pc) from " << pow(10.0, lg_pixel_size_start)/pc << " to " << pow(10.0, lg_pixel_size_stop)/pc << std::endl;
    for(auto jet_side : {true, false}) {

        // Ignore CJ
//        if(jet_side == false) {
//            continue;
//        }

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
        // Maximal step size. Will be used this value of geometrical path divided by n (which is smaller)
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

//            if(jet_side == true){
//                total_fluxes.push_back(total_flux);
//            }
            total_fluxes.push_back(total_flux);

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



std::vector<double> run_on_analytic_params_kh(double redshift, double los_angle_deg, double R_1_pc,
                                              double b_0, double m_b,
                                              double s, double gamma_min,
                                              double background_fraction,
                                              double Gamma_0, double Gamma_1, double betac_phi,
                                              int number_of_pixels_along, int number_of_pixels_across,
                                              double lg_pixel_size_mas_start, double lg_pixel_size_mas_stop,
                                              std::vector<double> scale_spirals,
                                              std::vector<double> spiral_width_frac,
                                              double phase_0, double lambda_0, double amp_0,
                                              double phase_1, double lambda_1, double amp_1,
                                              double phase_2, double lambda_2, double amp_2) {
    auto t1 = Clock::now();
    std::clock_t start;
    start = std::clock();

    double los_angle = los_angle_deg*M_PI/180.;
    // Observed frequencies in GHz
    std::vector<double> nu_observed_ghz{15.3, 8.4};
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
    double R_1pc = R_1_pc*pc;
    Parabaloid geometry(origin, direction, R_1pc, big_scale, true);

    // Setting V-field =================================================================================================
    VField* vfield;
//    vfield = new ConstParabolicVField(Gamma, &geometry, betac_phi);
    // Gamma(z) = Gamma_0 + Gamma_1*(z/1pc)^{0.5}
    vfield = new AccelParabolicVField(Gamma_0, Gamma_1, &geometry, betac_phi);


    // Setting B-field - in Lab frame ==================================================================================
    BKScalarBField bk_bfield(b_0, m_b, &geometry, nullptr, vfield, false);

    std::vector<VectorBField*> vbfields;
    std::vector<ScalarBField*> sbfields;
    sbfields.push_back(&bk_bfield);


    // Setting components of N-fields ==================================================================================
    PowerLaw kh_particles(s-0.6, gamma_min, "pairs", false);
    PowerLaw bkg_particles(s, gamma_min, "pairs", false);


//    double omega = 1.4E-10;
    double omega = 0.0;
    // TODO: Make accepting a list of B-fields!
    // Two Es modes
    EquipartitionKHNfield kh_Es_nfield(&bkg_particles, &bk_bfield, &geometry, nullptr,
                                       vfield, omega);
    kh_Es_nfield.set_background_fraction(background_fraction);
    kh_Es_nfield.set_spiral_width_frac({0.025, 0.025});
    kh_Es_nfield.set_spiral_scale({2.0, 2.0});
    kh_Es_nfield.set_spiral(phase_0, lambda_0 * pc, amp_0 * R_1pc);
    kh_Es_nfield.set_spiral(phase_1, lambda_1 * pc, amp_1 * R_1pc);
//    if(amp_2 > 0.0) {
//        kh_Es_nfield.set_spiral(phase_2, lambda_2 * pc, amp_2 * R_1pc);
//    }

    // Eb mode
    EquipartitionKHNfield kh_Eb_nfield(&kh_particles, &bk_bfield, &geometry, nullptr,
                                       vfield, omega);
    kh_Eb_nfield.set_background_fraction(background_fraction);
    kh_Eb_nfield.set_spiral_width_frac({0.05});
    kh_Eb_nfield.set_spiral_scale({3.0});
    kh_Eb_nfield.set_spiral(phase_2, lambda_2 * pc, amp_2 * R_1pc);


    std::vector<NField*> nfields;
    nfields.push_back(&kh_Es_nfield);
    nfields.push_back(&kh_Eb_nfield);


    EquipartitionBKNfield bkg_nfield(&bkg_particles, sbfields, &geometry, nullptr,
                                     vfield, 0.0, 0.005);
    nfields.push_back(&bkg_nfield);


    Jet bkjet(&geometry, vfield, sbfields, vbfields, nfields);

    // Setting parameters of pixels and image ==========================================================================
    double pixel_size_mas_start = pow(10.0, lg_pixel_size_mas_start);
    double pixel_size_mas_stop = pow(10.0, lg_pixel_size_mas_stop);

    // Non-uniform pixel from ``pixel_size_mas_start`` (near BH) to ``pixel_size_mas_stop`` (image edges)
    auto image_size = std::make_pair(number_of_pixels_across, number_of_pixels_along);
    auto pc_in_mas = mas_to_pc(redshift);
    std::cout << "pc_in_mas " << pc_in_mas << std::endl;
    // Log10 of pixel size in cm
    auto lg_pixel_size_start = log10(pixel_size_mas_start*pc_in_mas*pc);
    auto lg_pixel_size_stop = log10(pixel_size_mas_stop*pc_in_mas*pc);

    std::cout << "Setting pixel size (pc) from " << pow(10.0, lg_pixel_size_start)/pc << " to " << pow(10.0, lg_pixel_size_stop)/pc << std::endl;
    for(auto jet_side : {true, false}) {

        // Ignore CJ
//        if(jet_side == false) {
//            continue;
//        }

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
        // Maximal step size. Will be used this value of geometrical path divided by n (which is smaller)
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

//            if(jet_side == true){
//                total_fluxes.push_back(total_flux);
//            }
            std::cout << "Flux = " << total_flux << " Jy\n";
            total_fluxes.push_back(total_flux);

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



std::vector<double> run_on_analytic_params_t(double redshift, double los_angle_deg, double cone_half_angle_deg,
                                             double b_0, double m_b,
                                             double s, double gamma_min,
                                             double K_1, double n,
                                             double Gamma,
                                             int number_of_pixels_along, int number_of_pixels_across,
                                             double lg_pixel_size_mas_start, double lg_pixel_size_mas_stop,
                                             double t_obs_days,
                                             std::vector<double> flare_params) {

    bool adjust_high_freq_resolution = false;
    auto t1 = Clock::now();
    std::clock_t start;
    start = std::clock();

    auto pc_in_mas = mas_to_pc(redshift);
    std::cout << "pc_in_mas " << pc_in_mas << std::endl;

    double los_angle = los_angle_deg*M_PI/180.0;

    // Observed frequencies in GHz
    std::vector<double> nu_observed_ghz{2.3, 8.6};
    double min_nu_ghz = *std::min_element(nu_observed_ghz.begin(), nu_observed_ghz.end());
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

    // We need to set VField for flares first===========================================================================
    VField* vfield;
    bool central_vfield = true;
    if (central_vfield) {
        vfield = new ConstCentralVField(Gamma, &geometry, 0.0);
    } else {
        vfield = new ConstFlatVField(Gamma, &geometry, 0.00);
    }
    // =================================================================================================================

    // Setting B-field =================================================================================================
    BKScalarBField bk_bfield(b_0, m_b, &geometry);

    std::vector<VectorBField*> vbfields;
    std::vector<ScalarBField*> queiscent_sbfields;
    std::vector<ScalarBField*> flaring_sbfields;
    queiscent_sbfields.push_back(&bk_bfield);

    double frac_amp, frac_amp_B, t_start_days, flare_width_pc;
    int num_flares = flare_params.size()/4;
//    FlareBKScalarBField* bk_flare_bfield;
//    for(int i = 0; i < num_flares; i++){
//        frac_amp = flare_params[4*i + 0];
//        frac_amp_B = flare_params[4*i + 1];
//        t_start_days = flare_params[4*i + 2];
//        // In sec
//        t_start_days *= 24.0 * 60.0 * 60.0;
//        flare_width_pc = flare_params[4*i + 3];
//        bk_flare_bfield = new FlareBKScalarBField(frac_amp_B*b_0, m_b, t_start_days, flare_width_pc,
//                                                  los_angle, redshift, &geometry, nullptr, vfield);
//        queiscent_sbfields.push_back(bk_flare_bfield);
//    }

    // =================================================================================================================

    // Setting components of N-fields ==================================================================================
    PowerLaw particles(s, gamma_min, "pairs");
//    BKNField bk_stat_nfield(K_1, n, &particles, true, &geometry);
    EquipartitionBKNfield bk_stat_nfield(&particles, queiscent_sbfields, &geometry, nullptr, vfield);

    std::vector<NField*> queiscent_nfields;
    std::vector<NField*> flaring_nfields;
    queiscent_nfields.push_back(&bk_stat_nfield);
    flaring_nfields.push_back(&bk_stat_nfield);

    FlareBKNField* bk_flare_nfield;
    for(int i = 0; i < num_flares; i++){
        frac_amp = flare_params[4*i + 0];
        frac_amp_B = flare_params[4*i + 1];
        t_start_days = flare_params[4*i + 2];
        // In sec
        t_start_days *= 24.0 * 60.0 * 60.0;
        flare_width_pc = flare_params[4*i + 3];
        bk_flare_nfield = new FlareBKNField(&bk_stat_nfield, frac_amp, t_start_days, flare_width_pc, vfield);
        flaring_nfields.push_back(bk_flare_nfield);
    }


    // Quiscent jet
//    Jet bkjet(&geometry, vfield, queiscent_sbfields, vbfields, queiscent_nfields);
    // Flare N-only jet
    Jet bkjet(&geometry, vfield, queiscent_sbfields, vbfields, flaring_nfields);

//    // FIXME: Put inside frequency loop for dep. on frequency
//    // Setting parameters of pixels and image ==========================================================================
//    // Non-uniform pixel from ``pixel_size_mas_start`` (near BH) to ``pixel_size_mas_stop`` (image edges)
//    double pixel_size_mas_start = pow(10.0, lg_pixel_size_mas_start);
//    double pixel_size_mas_stop = pow(10.0, lg_pixel_size_mas_stop);
////    auto image_size = std::make_pair(number_of_pixels_across, number_of_pixels_along);
//    auto pc_in_mas = mas_to_pc(redshift);
//    std::cout << "pc_in_mas " << pc_in_mas << std::endl;
//    // Log10 of pixel size in cm
//    auto lg_pixel_size_start = log10(pixel_size_mas_start*pc_in_mas*pc);
//    auto lg_pixel_size_stop = log10(pixel_size_mas_stop*pc_in_mas*pc);

//    std::cout << "Setting pixel size (pc) from " << pow(10.0, lg_pixel_size_start)/pc << " to " << pow(10.0, lg_pixel_size_stop)/pc << std::endl;
    for(auto jet_side : {true, false}) {

        // Ignore CJ
        if(jet_side == false) {
            continue;
        }

//        ImagePlane imagePlane(image_size, lg_pixel_size_start, lg_pixel_size_stop, los_angle, jet_side);
//        // Array of pixel sizes in cm
//        auto pixel_sizes = imagePlane.getPixelSizes();
//        // Array of pixel solid angles in rad*rad
//        std::vector<std::vector<double>> pixel_solid_angles;
//        pixel_solid_angles.resize(pixel_sizes.size());
//
//        for(unsigned long i=0; i < pixel_sizes.size(); i++) {
//            pixel_solid_angles[i].resize(pixel_sizes[0].size());
//            for(unsigned long j=0; j < pixel_sizes[0].size(); j++) {
//                // Divide by ``pc_in_mas*pc`` to bring ``cm`` to ``mas`` at source redshift
//                pixel_solid_angles[i][j] = (pixel_sizes[i][j]/(pc_in_mas*pc))*(pixel_sizes[i][j]/(pc_in_mas*pc))*mas_to_rad*mas_to_rad;
//            }
//        }
//
//        // Array of scale factors. Divide resulting image on this to obtain flux density in Jy. Accounts for cosmological
//        // scaling of intensity
//        std::vector<std::vector<double>> scales;
//        scales.resize(pixel_sizes.size());
//        for(unsigned long i=0; i < pixel_sizes.size(); i++) {
//            scales[i].resize(pixel_sizes[0].size());
//            for(unsigned long j=0; j < pixel_sizes[0].size(); j++) {
//                scales[i][j] = 1E-23*(1.+redshift)*(1.+redshift)*(1.+redshift)/pixel_solid_angles[i][j];
//            }
//        }
//
//        Observation observation(&bkjet, &imagePlane);

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

            // Setting parameters of pixels and image ==================================================================
            double pixel_size_mas_start, pixel_size_mas_stop;
            std::pair<unsigned long int, unsigned long int> image_size;
            if(adjust_high_freq_resolution) {
                std::cout << "Image size : along = " << number_of_pixels_along * std::round(nu_observed_ghz[i_nu] / min_nu_ghz)
                          << ", acrosss = " << number_of_pixels_across * std::round(nu_observed_ghz[i_nu] / min_nu_ghz)
                          << "\n";
                std::cout << "Pixel logsize (mas) start = "
                          << lg_pixel_size_mas_start - log10(nu_observed_ghz[i_nu] / min_nu_ghz) << ", stop = "
                          << lg_pixel_size_mas_stop - log10(nu_observed_ghz[i_nu] / min_nu_ghz) << "\n";
                // Non-uniform pixel from ``pixel_size_mas_start`` (near BH) to ``pixel_size_mas_stop`` (image edges)
                pixel_size_mas_start = pow(10.0, lg_pixel_size_mas_start - log10(nu_observed_ghz[i_nu]/min_nu_ghz));
                pixel_size_mas_stop = pow(10.0, lg_pixel_size_mas_stop - log10(nu_observed_ghz[i_nu]/min_nu_ghz));
                image_size = std::make_pair(number_of_pixels_across * std::round(nu_observed_ghz[i_nu]/min_nu_ghz),
                                            number_of_pixels_along * std::round(nu_observed_ghz[i_nu]/min_nu_ghz));
            }
            else {
                std::cout << "Image size : along = " << number_of_pixels_along
                          << ", acrosss = " << number_of_pixels_across
                          << "\n";
                std::cout << "Pixel logsize (mas) start = "
                          << lg_pixel_size_mas_start << ", stop = "
                          << lg_pixel_size_mas_stop << "\n";
                // Non-uniform pixel from ``pixel_size_mas_start`` (near BH) to ``pixel_size_mas_stop`` (image edges)
                pixel_size_mas_start = pow(10.0, lg_pixel_size_mas_start);
                pixel_size_mas_stop = pow(10.0, lg_pixel_size_mas_stop);
                image_size = std::make_pair(number_of_pixels_across,
                                            number_of_pixels_along);
            }

            // Log10 of pixel size in cm
            auto lg_pixel_size_start = log10(pixel_size_mas_start*pc_in_mas*pc);
            auto lg_pixel_size_stop = log10(pixel_size_mas_stop*pc_in_mas*pc);


            ImagePlane imagePlane(image_size,lg_pixel_size_start,lg_pixel_size_stop, los_angle, jet_side);
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


            // t_obs_days in days - need to move to sec.
            observation.observe(n_, tau_max, dt_max, tau_min, nu_bh[i_nu], polarization, relerr,
                                24.0 * 60.0 * 60.0 * t_obs_days);
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
            ss << std::fixed << std::setprecision(1) << std::showpoint << t_obs_days;
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

//// To quickly show J + CJ images for freq_ghz GHz:
//// j = np.loadtxt("jet_image_i_{}.txt".format(freq_ghz)); cj = np.loadtxt("cjet_image_i_{}.txt".format(freq_ghz)); jcj = np.hstack((cj[::, ::-1], j)); plt.matshow(jcj, aspect="auto");plt.colorbar(); plt.show()
//int main() {
//
//    std::vector<double> total_fluxes;
//    total_fluxes = run_on_analytic();
//    for(auto total_flux: total_fluxes){
//        std::cout << "Total flux [Jy] = " << total_flux << "\n";
//    }
//    return 0;
//}


// To run in parallel when file params_3ridges.txt has 3 parameter sets:
// parallel --files --results t_obs_{11} --joblog log --jobs 7 -a params_flaring_jet.txt -n 1 -m --colsep ' ' "./bk_transfer"
int main_t(int argc, char *argv[]) {

    std::vector<double> total_fluxes;
    std::vector<double> flare_params;

    int num_of_flares = (argc - 12) / 4;
    std::cout << "Number of flares : " << num_of_flares << "\n";
    int argc_pred = 12 + 4 * num_of_flares;

    if(argc != argc_pred){
        std::cout << argc << "\n";
        std::cout << "Supply redshift, LOS-angle (deg), Cone half-angle (deg),"
                     " B_1 [G], K_1 [cm^{-3}], Gamma, N_along, N_across,"
                     " lg_pixel_size_mas_start, lg_pixel_size_mas_stop,\n"
                     " t_obs (days), flare_params: [amp_N, amp_B, t_start (month), width (pc)]\n" << "\n";
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
        std::cout << "t_obs(days) = " << argv[11] << "\n";

        double frac_amp, frac_amp_B, t_start_days, flare_width_pc;
        for(int i = 0; i < num_of_flares; i++){
            frac_amp = atof(argv[11 + 4*i + 1]);
            std::cout << "Frac.amp N = " << frac_amp << "\n";
            frac_amp_B = atof(argv[11 + 4*i + 2]);
            std::cout << "Frac.amp B = " << frac_amp_B << "\n";
            t_start_days = atof(argv[11 + 4*i + 3]);
            std::cout << "T_start (days) = " << t_start_days << "\n";
            flare_width_pc = atof(argv[11 + 4*i + 4]);
            std::cout << "Width (pc) = " << flare_width_pc << "\n";
            flare_params.push_back(frac_amp);
            flare_params.push_back(frac_amp_B);
            flare_params.push_back(t_start_days);
            flare_params.push_back(flare_width_pc);
        }

        // FIXME: 4 debug run
//        flare_params.push_back(5.0);
//        flare_params.push_back(0.0);
//        flare_params.push_back(0.0);
//        flare_params.push_back(0.2);
//        double redshift = 1.0;
//        double los_angle_deg = 5.0;
//        double cone_half_angle_deg = 1.0;
//        double b_0 = 0.85;
//        double K_1 = 500.;
//        double Gamma = 8.6;
//        int number_of_pixels_along = 400;
//        int number_of_pixels_across = 80;
//        double lg_pixel_size_mas_start = -2.5;
//        double lg_pixel_size_mas_stop = -0.5;
//        double t_obs = 1000.;

        total_fluxes = run_on_analytic_params_t(redshift, los_angle_deg, cone_half_angle_deg,
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


int main(int argc, char *argv[]) {

    std::vector<double> total_fluxes;

//    double s = 2.0;
//    double gamma_min = 10.0;
//    double background_fraction = 0.01;

//    double los_angle_deg = atof(argv[1]);
//    std::cout << "LOS (deg) = " << argv[1] << "\n";
//
//    double R_1_pc = atof(argv[2]);
//    std::cout << "R_1_pc = " << argv[2] << "\n";
//
//    double b_0 = atof(argv[3]);
//    std::cout << "b_0 = " << argv[3] << "\n";
//
//    double m_b = atof(argv[4]);
//    std::cout << "m_b = " << argv[4] << "\n";
//
//    double phase_0 = atof(argv[5]);
//    std::cout << "phase_0 = " << argv[5] << "\n";
//
//    double lambda_0 = atof(argv[6]);
//    std::cout << "lambda_0 = " << argv[6] << "\n";
//
//    double phase_1 = atof(argv[7]);
//    std::cout << "phase_1 = " << argv[7] << "\n";
//
//    double lambda_1 = atof(argv[8]);
//    std::cout << "lambda_1 = " << argv[8] << "\n";


    double los_angle_deg = 17.;
    double s = 2.0;
    double gamma_min = 10.0;
    double background_fraction = 0.0001;

//    double R_1_pc = 0.13;
    double R_1_pc = 0.105;
    double b_0 = 0.12;
    double m_b = 0.5;

    double Gamma_0 = 1.1;
    double Gamma_1 = 0.5;
    double betac_phi = 0.0;

    std::vector<double> scale_spirals = {2.0, 2.0, 3.0};
    std::vector<double> spiral_width_frac = {0.025, 0.025, 0.05};

    // R = 0.09367584
    double phase_0 = 80.0*M_PI/180.;
    double lambda_0 = 8.02;
    double amp_0 = 0.72;

    // The Es mode that goes from down to up and crosses other Es mode at z_obs = 25 mas
    // R = 0.12311555
    double phase_1 = 212.0*M_PI/180.;
    double lambda_1 = 7.61;
    double amp_1 = 0.95;

    // Eb mode
    // R = 0.05380604
    double phase_2 = 66.*M_PI/180.;
    double lambda_2 = 2.57;
    double amp_2 = 0.41;

    total_fluxes = run_on_analytic_params_kh(0.00436, los_angle_deg, R_1_pc, b_0, m_b, s, gamma_min,
                                             background_fraction, Gamma_0, Gamma_1, betac_phi, 350,
                                             100, -2.0, -0.5,
                                             scale_spirals, spiral_width_frac,
                                             phase_0, lambda_0, amp_0,
                                             phase_1, lambda_1, amp_1,
                                             phase_2, lambda_2, amp_2);
    for(auto total_flux: total_fluxes){
        std::cout << "Total flux [Jy] = " << total_flux << "\n";
    }

    return 0;
}