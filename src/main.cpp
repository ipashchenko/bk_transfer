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
    double redshift = 0.00436;
//    double redshift = 0.1;
    double los_angle = 17.0*M_PI/180.0;
    // theta_pr = 60 deg for Gamma = 3
//    double los_angle = 11.3*M_PI/180.0;

    // FIXME: tested
//    double los_angle = 0.75*M_PI/180.0;


    //double redshift = 0.01;
    // theta_pr = 60 deg for Gamma = 3
    //double los_angle = 11.3*M_PI/180.0;
    //double los_angle = 30.0*M_PI/180.0;
    // Observer frame angle for plasma frame angle 30deg and Gamma = 3
    //double los_angle = 5.3*M_PI/180.0;
    // Observer frame angle for plasma frame angle 10deg and Gamma = 3
    //double los_angle = 1.7*M_PI/180.0;
    //double los_angle = 3.0*M_PI/180.0;

    // Observed frequencies in GHz
    //std::vector<double> nu_observed_ghz{15.4, 12.1, 8.1};
//    std::vector<double> nu_observed_ghz{15.4, 8.1};
    std::vector<double> nu_observed_ghz{15.4};
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
    double cone_half_angle = 2.0*M_PI/180.0;
    // For sheath
    //double R = 0.15*pc;
    // For jet only
    double R_1pc = 0.1*pc;
//    Cone geometry(origin, direction, cone_half_angle, big_scale);
//    Cylinder geometry(origin, direction, 0.125*pc);
    Parabaloid geometry(origin, direction, R_1pc, big_scale, true);

    // Setting B-field
    // Value at r=1pc
    //double B_1 = 0.025;
    //double B_1_jet = 0.025;
    //double B_1_sheath = 0.025;
    //double B_1_spine = 0.01;
    // Exponent of the decrease
    //double m = 1.0;
    //double pitch_angle = 1.55;
    //double pitch_angle = 45.0*M_PI/180.;
    //double pitch_angle_sheath = 45.0*M_PI/180.;
    //double pitch_angle_spine = 10.0*M_PI/180.;
    //ConstCylinderBFieldZ bfield(B_1, m, true);
    //RadialConicalBField bfield(B_1, m, false);
    //ToroidalBField bfield(B_1, m, true);
    //SpiralConicalBField bfield(B_1, pitch_angle, true);
    //HelicalCylinderBField bfield(B_1, pitch_angle, true, 0.0, geometry_in);


    // Setting geometry of Faraday-rotating screen
    //double cone_half_angle_in = 2.0*M_PI/180.0;
    //double cone_half_angle_out = 3.0*M_PI/180.0;
    //double R_in = 0.117*pc;
//    double R_spine = 0.09*pc;
    //double R_out = 0.15*pc;
    //Cone geometry_in(origin, direction, cone_half_angle_in, big_scale);
    //Cone geometry_out(origin, direction, cone_half_angle_out, big_scale);
    // Spine radius
//    Cylinder geometry_spine(origin, direction, R_spine);
    //Parabaloid geometry_spine(origin, direction, 0.025*pc, big_scale);
    // Sheath outer radius
//    Cylinder geometry_sheath(origin, direction, R);
    //Parabaloid geometry_sheath(origin, direction, 0.1*pc, big_scale);
    // Outer sheath
    //Cylinder geometry_out(origin, direction, R_out);


    // Setting components of B-fields ==================================================================================

//    HelicalCylinderBField jetbfield(0.1, 60.0*M_PI/180., true, 0.0, &geometry);
//    ConstCylinderBFieldZ jetbfield(0.1, 1.0, false, 0.0, &geometry);
//    ToroidalBField jetbfield(0.1, 1.0, false, 0.0, &geometry);


    // Jet B-field (inside inner cylinder)
//    ConstCylinderBFieldZ jetbfield(0.01, 1, true, 0.0, &geometry);

    // TODO: Tested
    HelicalConicalBField jetbfield(0.01, 0.5, 10.*M_PI/180., true, 0.0, &geometry);
//    HelicalCylinderBField jetbfield(0.0001, 30.0*M_PI/180., true, 0.0, &geometry);
//    ToroidalBField jetbfield(0.01, 1.0, false, 0.0, &geometry);

    //HelicalCylinderBField jetbfield_sheath(B_1_sheath, pitch_angle_sheath, true, 0.0,
    //                                       &geometry_in, &geometry_inner);
    //HelicalCylinderBField jetbfield_spine(B_1_spine, pitch_angle_spine, true, 0.0,
    //                                      &geometry_in);
    //ReversedPinchCylindricalBField jetbfield(0.1, 0.0, &geometry);
//    ReversedPinchConicalBField jetbfield(0.1, 1, &geometry, 0.0);

    // Faraday-rotating screen B-field (between inner and outer cylinders)
    //double pitch_angle = 45.0*M_PI/180.;
    //HelicalConicalBField rotm_screen_bfield(B_1, m, pitch_angle, true, 0.0, &geometry_out, &geometry_in);
    //HelicalCylinderBField rotm_screen_bfield(B_1_sheath, pitch_angle, true, 0.0, &geometry_out, &geometry_in);
    //ToroidalBField rotm_screen_bfield(B_1_sheath, 0.0, true, 0.0, &geometry_out, &geometry_in);

    // Lena MHD
    //LenaMHDBField jetvbfield(0.0, 0.0, 0.0);
    // Check
//    ToroidalBField jetbfield(0.05, 1, false, 0.0, &geometry);
    //ConstCylinderBFieldZ jetbfield(0.01, 0, true, 0.0, &geometry);

    // Lena MHD shells
    //LenaMHDBFieldShell bfield(0.0, -0.02, 0.1, 0.15, 0.0);

    std::vector<VectorBField*> vbfields;
    vbfields.push_back(&jetbfield);
    //bfields.push_back(&jetbfield_sheath);
    //bfields.push_back(&jetbfield_spine);
    //bfields.push_back(&rotm_screen_bfield);

    // Setting components of N-fields ==================================================================================
    // Exponent of the power-law particle Lorenz factor distribution
    double s = 2.5;
    double ds = 0.0;
    double gamma_min = 1.0;
    PowerLaw particles(s, gamma_min, "pairs");
    // Value at r=1pc
    double K_1 = 500;
    //double K_1_spine = 50;
    //double K_1_sheath = 600;
    // Exponent of the decrease
    double n = 1.5;
    // Exponent of the power-law particle Lorenz factor distribution
    //double s = 2.5;
    //double ds = 0.0;
    //BKNField power_law_nfield_spine(0.0175, 1, true, 2.1, 150, &geometry_spine,
    //                                "pairs", false);

    // TODO: Tested
    BKNField power_law_nfield_spine(K_1, n, &particles, true, &geometry);
    power_law_nfield_spine.set_spiral(0.0, 30.0*R_1pc, 0.9*R_1pc);
    power_law_nfield_spine.set_spiral(M_PI, 30.0*R_1pc, 0.9*R_1pc);
    power_law_nfield_spine.set_spiral(M_PI/6.0, 10.0*R_1pc, 0.5*R_1pc);
//    power_law_nfield_spine.set_heating_profile(0.95, 0.01, 0.01);

    //SheathPowerLawNField power_law_nfield_sheath(2*0.875, 1, true, 2.1, 150,
    //                                             &geometry_spine, &geometry_sheath, "pairs", false);
//    SheathPowerLawNField power_law_nfield_sheath(0.5, 0, true, 2.2, 200,
//                                                 &geometry_spine, &geometry_sheath, "pairs", false);
    //BKNField power_law_nfield_spine(K_1_spine, n, true, s, 100, &geometry_inner, "normal", ds);
    //SheathPowerLawNField power_law_nfield_sheath(K_1_sheath, n, true, s, 100,
    //                                             &geometry_inner, &geometry_in, "normal", ds);
    // MHD profile
    //LenaMHDNField power_law_nfield(2.5, 10, &geometry, "pairs", true);
    //power_law_nfield.set_profile(0.01, 0.3, 7.0, 3.0, 2.0, 0.3, 0.1);
    // Edges
    //power_law_nfield.set_profile(0.3, 0.7, 7.0, 7.0, 0.1, 0.3, 2);
    //ShearBKNField nfield(0.5*cone_half_angle, 0.0, 1.0, K_1, n, true, s, ds);
    //SheathBKNField nfield(0.5*cone_half_angle, K_1, n, false, s, ds);

    // Thermal plasma density profile in Faraday-rotating screen
    //double n_0_rm = 100;
    //double n_n_rm = 0.0;
    //SheathThermalNField rotm_screen_nfield(n_0_rm, n_n_rm, &geometry_in, &geometry_out);

    // Put two density profiles in a vector
    //LenaMHDNField power_law_nfield(2.5, 150, &geometry, "pairs", false);
    //power_law_nfield.set_profile(0.3, 0.7, 7.0, 7.0, 0.3, 0.3, 1);
    std::vector<NField*> nfields;
    nfields.push_back(&power_law_nfield_spine);
//    nfields.push_back(&power_law_nfield_sheath);
    //nfields.push_back(&power_law_nfield_spine);
    //nfields.push_back(&power_law_nfield_sheath);
    //nfields.push_back(&rotm_screen_nfield);

    // Lena MHD
    //LenaMHDNField nfield(s, 10, &geometry, "normal");
    //nfield.set_profile(0.0, 0.8, 1.0, 10.0, 0.0, 0.0, 1.0);

    // Lena MHD shell
    //LenaMHDNFieldShell nfield(800.0, s, 0.1, 0.15);

    // Setting V-field =================================================================================================
    VField* vfield;
    bool central_vfield = false;
//    double Gamma = 3.42;
    double Gamma = 1.20;
    if (central_vfield) {
        vfield = new ConstCentralVField(Gamma, &geometry, 0.0);
    } else {
        vfield = new ConstFlatVField(Gamma, &geometry, 0.05);
    }
    //vfield = new ConstFlatVField(Gamma, &geometry, 0.0);
    //vfield = new SheathFlatVField(2.0, 3.4, &geometry_spine, &geometry_sheath,
    //                              0.0, 0.0, -0.1, -0.1);
//    vfield = new SheathFlatVField(2.0, 4.0, &geometry_spine, &geometry_sheath,
//                                  0.0, 0.0, 0.0, 0.0);
    //vfield = new SheathFlatVField(1.0, 1.0, R_in);
    //vfield = new SheathCentralVField(3.0, 1.4, cone_half_angle);
    //vfield = new ShearedCentralVField(2.0, 15.0, cone_half_angle);

    Jet bkjet(&geometry, vfield, vbfields, nfields);

    // FIXME: Put inside frequency loop for dep. on frequency
    // Setting parameters of pixels and image ==========================================================================
    int number_of_pixels_along = 512;
//    int number_of_pixels_along = 500;
    //int number_of_pixels_across = 150;
    int number_of_pixels_across = 64;
    // Non-uniform pixel from ``pixel_size_mas_start`` (near BH) to ``pixel_size_mas_stop`` (image edges)
    double pixel_size_mas_start = pow(10.0, -1.5);
    //double pixel_size_mas_start = 0.06;
    double pixel_size_mas_stop = pow(10.0, -0.5);
    //double pixel_size_mas_stop = 0.06;
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
        double tau_max = 10;
        double dt_max_pc = 0.01;
        double dt_max = pc*dt_max_pc;
        double tau_min_log10 = -20.0;
        double tau_min = pow(10.,tau_min_log10);
        int n_ = 100;
        double relerr = 1e-10;

        // Solve for all Stokes parameters ("full") or only full intensity ("I")?
//        string polarization = "I";
        string polarization = "full";

        for(int i_nu=0; i_nu < nu_observed_ghz.size(); i_nu++) {
            if(jet_side) {
                std::cout << "Running transfer for frequency " << nu_observed_ghz[i_nu] << " GHz for approaching jet" << std::endl;
            } else {
                std::cout << "Running transfer for frequency " << nu_observed_ghz[i_nu] << " GHz for counter-jet" << std::endl;
            }
            observation.run(n_, tau_max, dt_max, tau_min, nu_bh[i_nu], polarization, relerr);
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

int main() {
    std::vector<double> total_fluxes;
    total_fluxes = run_on_analytic();
    for(auto total_flux: total_fluxes){
        std::cout << "Total flux [Jy] = " << total_flux << "\n";
    }
    return 0;
}