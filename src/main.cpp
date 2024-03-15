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


std::vector<double> run_on_analytic_params_t(double redshift, double los_angle_deg, double R_cyl_pc,
                                             double b_0, double m_b,
                                             double s, double gamma_min,
                                             double Gamma,
                                             int number_of_pixels_along, int number_of_pixels_across,
                                             double lg_pixel_size_mas_start, double lg_pixel_size_mas_stop) {

    bool adjust_high_freq_resolution = false;
    auto t1 = Clock::now();
    std::clock_t start;
    start = std::clock();

    auto pc_in_mas = mas_to_pc(redshift);
    std::cout << "pc_in_mas " << pc_in_mas << std::endl;

    double los_angle = los_angle_deg*M_PI/180.0;

    // Observed frequencies in GHz
//	std::vector<double> nu_observed_ghz{2.3, 8.6};
	std::vector<double> nu_observed_ghz{15.4};
    double min_nu_ghz = *std::min_element(nu_observed_ghz.begin(), nu_observed_ghz.end());
//	std::vector<std::string> nu_observed_band{"S", "X"};
	std::vector<std::string> nu_observed_band{"u"};
    std::vector<double> total_fluxes;
    // Frequencies in the BH frame in Hz
    std::vector<double> nu_bh;
    for(auto nu_obs_ghz : nu_observed_ghz) {
        nu_bh.push_back(nu_obs_ghz*1E+09*(1.0+redshift));
    }

    // Setting geometry ================================================================================================
    Vector3d origin = {0., 0., 0.};
    Vector3d direction = {0., 0., 1.};
//	Cone geometry(origin, direction, los_angle/2., 100*pc);
    Cylinder geometry(origin, direction, R_cyl_pc*pc);
//    Cylinder geometry_out(origin, direction, R_cyl_pc);
//    Cylinder geometry_in(origin, direction, R_cyl_pc);
    
    // We need to set VField for flares first===========================================================================
    VField* vfield;
    bool central_vfield = false;
    if (central_vfield) {
        vfield = new ConstCentralVField(Gamma, &geometry, 0.0);
    } else {
        vfield = new ConstFlatVField(Gamma, &geometry, 0.00);
    }
    // =================================================================================================================

    // Setting B-field =================================================================================================
    std::vector<VectorBField*> vbfields;
    std::vector<ScalarBField*> sbfields;
	
	// We need polarization!
    ToroidalBField vbfield(b_0, m_b, true, 0.0, &geometry, nullptr);
	// HelicalCylinderBField vbfield(b_0, 89.*M_PI/180., true, 0.0, &geometry_out, nullptr);
	vbfields.push_back(&vbfield);
    // =================================================================================================================

    // Setting emitting components of N-fields ==================================================================================
    PowerLaw particles(s, gamma_min, "normal");
    // FIXME: When calculating full magnetic field in plasma frame - sum fields as vectors!
    EquipartitionBKNfield nfield(&particles, sbfields, vbfields, &geometry, nullptr, vfield);

	
	std::vector<NField*> nfields;
    nfields.push_back(&nfield);
    // =================================================================================================================

    // Flare N-only jet
    Jet bkjet(&geometry, vfield, sbfields, vbfields, nfields);

    // Setting transfer-specific parameters ========================================================================
    double tau_max = 30;
    double dt_max_pc = 0.01;
    double dt_max = pc*dt_max_pc;
    double tau_min_log10 = -20.0;
    double tau_min = pow(10.,tau_min_log10);
    int n_ = 100;
    double relerr = 1e-10;

    for(auto jet_side : {true, false}) {

        // Ignore CJ
        if(jet_side == false) {
            continue;
        }

        // Solve for all Stokes parameters ("full") or only full intensity ("I")?
//        string polarization = "I";
        string polarization = "full";

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
            std::stringstream ss;
            std::string freq_name = nu_observed_band[i_nu];

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

// To quickly show J + CJ images for freq_ghz GHz:
// j = np.loadtxt("jet_image_i_{}.txt".format(freq_ghz)); cj = np.loadtxt("cjet_image_i_{}.txt".format(freq_ghz)); jcj = np.hstack((cj[::, ::-1], j)); plt.matshow(jcj, aspect="auto");plt.colorbar(); plt.show()
int main(int argc, char *argv[]) {

    std::vector<double> total_fluxes;
    std::vector<double> flare_params;
	double redshift, los_angle_deg, R_cyl_pc, b_0, m_b, Gamma, lg_pixel_size_mas_start, lg_pixel_size_mas_stop;
	int number_of_pixels_along, number_of_pixels_across;

    int argc_pred = 11;

    if(argc != argc_pred){
        std::cout << argc << "\n";
        std::cout << "Supply redshift, LOS-angle (deg), R_cyl (pc),"
                     " B_1 [G], m_b, Gamma, N_along, N_across,"
                     " lg_pixel_size_mas_start, lg_pixel_size_mas_stop\n" << "\n";
        return 1;
    }
    else {
        redshift = atof(argv[1]);
        std::cout << "z = " << argv[1] << "\n";

		los_angle_deg = atof(argv[2]);
        std::cout << "LOS (deg) = " << argv[2] << "\n";

        R_cyl_pc = atof(argv[3]);
        std::cout << "R_cyl (pc) = " << argv[3] << "\n";

        b_0 = atof(argv[4]);
        std::cout << "B_1 (G) = " << argv[4] << "\n";
	
		m_b = atof(argv[5]);
		std::cout << "m_b = " << argv[5] << "\n";
		
        Gamma = atof(argv[6]);
        std::cout << "Gamma = " << argv[6] << "\n";

		number_of_pixels_along = atoi(argv[7]);
        std::cout << "N_along = " << argv[7] << "\n";

        number_of_pixels_across = atoi(argv[8]);
        std::cout << "N_across = " << argv[8] << "\n";

        lg_pixel_size_mas_start = atof(argv[9]);
        std::cout << "lg(min_pixel_size[mas]) = " << argv[9] << "\n";

        lg_pixel_size_mas_stop = atof(argv[10]);
        std::cout << "lg(max_pixel_size[mas]) = " << argv[10] << "\n";
        }

        // FIXME: 4 debug run
//        double redshift = 0.1;
//        double los_angle_deg = 90.0;
//		double R_cyl_pc = 0.7;
//        double b_0 = 0.1;
//		double m = 0.0;
//        double Gamma = 1.1;
//        int number_of_pixels_along = 200;
//        int number_of_pixels_across = 100;
//        double lg_pixel_size_mas_start = -2.;
//        double lg_pixel_size_mas_stop = -2.;
//		double s = 2.5;
//		double gamma_min = 10.0;

        total_fluxes = run_on_analytic_params_t(redshift, los_angle_deg, R_cyl_pc,
                                                b_0, m_b,
                                                2.5, 10.0,
                                                Gamma,
                                                number_of_pixels_along, number_of_pixels_across,
                                                lg_pixel_size_mas_start, lg_pixel_size_mas_stop);
    for(auto total_flux: total_fluxes){
        std::cout << "Total flux [Jy] = " << total_flux << "\n";
    }

    return 0;
}

