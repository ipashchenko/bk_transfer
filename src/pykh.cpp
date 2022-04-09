#include <string>
#include <iostream>
#include "ImagePlane.h"
#include "Observation.h"
#include "BField.h"
#include "VField.h"
#include "Cone.h"
#include "Parabaloid.h"
#include "Jet.h"
#include "utils.h"
#include <cmath>
#include "NField.h"
#include "Pixel.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using Eigen::Vector3d;
using Eigen::Matrix3Xd;
using std::vector;
using std::pair;
using namespace boost::numeric::odeint;
namespace ph = std::placeholders;


// Image must be scaled by ((pixsize_array/min(pixsize_array))**2).T to convert to intensity (
std::pair<vector<vector<double>>, vector<vector<double>>> run_on_analytic_params_kh(double redshift, double los_angle_deg, double R_1_pc,
                                                                                    double b_0,
                                                                                    double m_b,
                                                                                    double K_1, double n,
                                                                                    double s, double gamma_min,
                                                                                    double background_fraction,
                                                                                    double Gamma_0, double Gamma_1, double betac_phi,
                                                                                    int number_of_pixels_along, int number_of_pixels_across,
                                                                                    double lg_pixel_size_mas_start, double lg_pixel_size_mas_stop,
                                                                                    double spiral_width_frac,
                                                                                    double phase_0, double lambda_0, double amp_0,
                                                                                    double phase_1, double lambda_1, double amp_1,
                                                                                    double phase_2, double lambda_2, double amp_2,
                                                                                    double t_obs_days,
                                                                                    bool jet_side) {

    double los_angle = los_angle_deg*M_PI/180.;
    // Observed frequencies in GHz
    double nu_observed_ghz = 15.3;
//    double nu_observed_ghz = 8.4;
    // Frequencies in the BH frame in Hz
    double nu_bh = nu_observed_ghz*1E+09*(1+redshift);
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


    // Setting B-field - in Lab frame
    BKScalarBField bk_bfield(b_0, m_b, &geometry, nullptr, vfield, false);

    std::vector<VectorBField*> vbfields;
    std::vector<ScalarBField*> sbfields;
    sbfields.push_back(&bk_bfield);
//    vbfields.push_back(&jetbfield);



    // Setting components of N-fields ==================================================================================
    PowerLaw kh_particles(s, gamma_min, "pairs", false);

//    double omega = 1.4E-10;
    double omega = 0.0;
    EquipartitionKHNfield kh_nfield(&kh_particles, &bk_bfield, &geometry, nullptr,
                                    vfield, omega);
    kh_nfield.set_background_fraction(background_fraction);
    kh_nfield.set_spiral_width_frac(spiral_width_frac);
//    bk_stat_nfield.set_spiral(phase_0, lambda_0 * R_1pc, amp_0 * R_1pc);
//    bk_stat_nfield.set_spiral(phase_1, lambda_1 * R_1pc, amp_1 * R_1pc);
//    if(amp_2 > 0.0) {
//        bk_stat_nfield.set_spiral(phase_2, lambda_2 * R_1pc, amp_2 * R_1pc);
//    }
    kh_nfield.set_spiral(phase_0, lambda_0 * pc, amp_0 * R_1pc);
    kh_nfield.set_spiral(phase_1, lambda_1 * pc, amp_1 * R_1pc);
    if(amp_2 > 0.0) {
        kh_nfield.set_spiral(phase_2, lambda_2 * pc, amp_2 * R_1pc);
    }
    std::vector<NField*> nfields;
    nfields.push_back(&kh_nfield);


//    PowerLaw bkg_particles(s-0.3, gamma_min, "pairs", false);
//    EquipartitionBKNfield bkg_nfield(&bkg_particles, sbfields, &geometry, nullptr, vfield);
//    nfields.push_back(&bkg_nfield);


    Jet bkjet(&geometry, vfield, sbfields, vbfields, nfields);

    // Setting parameters of pixels and image ==========================================================================
    double pixel_size_mas_start = pow(10.0, lg_pixel_size_mas_start);
    double pixel_size_mas_stop = pow(10.0, lg_pixel_size_mas_stop);

    // Non-uniform pixel from ``pixel_size_mas_start`` (near BH) to ``pixel_size_mas_stop`` (image edges)
    auto image_size = std::make_pair(number_of_pixels_across, number_of_pixels_along);
    auto pc_in_mas = mas_to_pc(redshift);
//    std::cout << "pc_in_mas " << pc_in_mas << std::endl;
    // Log10 of pixel size in cm
    auto lg_pixel_size_start = log10(pixel_size_mas_start*pc_in_mas*pc);
    auto lg_pixel_size_stop = log10(pixel_size_mas_stop*pc_in_mas*pc);

//    std::cout << "Setting pixel size (pc) from " << pow(10.0, lg_pixel_size_start)/pc << " to " << pow(10.0, lg_pixel_size_stop)/pc << std::endl;

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

//    if(jet_side) {
//        std::cout << "Running transfer for frequency " << nu_observed_ghz << " GHz for approaching jet" << std::endl;
//    } else {
//        std::cout << "Running transfer for frequency " << nu_observed_ghz << " GHz for counter-jet" << std::endl;
//    }
    observation.observe(n_, tau_max, dt_max, tau_min, nu_bh, polarization, relerr, 24.0*60.0*60.0*t_obs_days);
    string value = "tau";
    auto image_tau = observation.getImage(value);

    value = "I";
    auto image_i = observation.getImage(value);
    for (unsigned long int i = 0; i < image_i.size(); ++i) {
        for (unsigned long int j = 0; j < image_i[i].size(); ++j) {
            image_i[i][j] = image_i[i][j]/scales[i][j];
        }
    }

    std::pair<vector<vector<double>>, vector<vector<double>>> result = std::make_pair(image_i, image_tau);
    return result;
}


// Compile with:
// c++ -O3 -Wall -shared -std=c++14 -L/usr/lib/openmpi -lmpi_cxx -fopenmp -fPIC -march=native -DNDEBUG -O3 -fext-numeric-literals `python3 -m pybind11 --includes` -o pykh`python3-config --extension-suffix` src/pykh.cpp src/BField.cpp src/Parabaloid.cpp src/Geometry.cpp src/Ray.cpp src/Intersection.cpp src/Pixel.cpp src/Image.cpp src/ImagePlane.cpp src/ParticlesDistribution.cpp src/NField.cpp src/VField.cpp src/Jet.cpp src/utils.cpp src/System.cpp src/Observation.cpp -I/home/ilya/github/bk_transfer/include  -I/usr/include/eigen3
PYBIND11_MODULE(pykh, m) {
    using namespace pybind11::literals; // for _a literal to define arguments
    m.doc() = "Radiative transfer for BK models"; // optional module docstring

//    m.def("run_on_analytics_params_kh", &run_on_analytic_params_kh, "Obtain Stokes I image with random B-field",
//          "los_angle"_a,
//          "redshift"_a,
//          "number_of_pixels_along"_a,
//          "number_of_pixels_across"_a,
//          "pixel_size_mas_start"_a,
//          "pixel_size_mas_stop"_a,
//          "cone_half_angle"_a,
//          "B_1"_a,
//          "m"_a,
//          "K_1"_a,
//          "n"_a,
//          "s"_a,
//          "ds"_a,
//          "Gamma"_a,
//          "nu_observed_ghz"_a,
//          "lg_tau_max"_a=4.,
//          "lg_tau_min"_a=-10.,
//          "dl_max_pc"_a=0.1,
//          "n_initial"_a=100,
//          "central_vfield"_a=false);


        m.def("run_on_analytic_params_kh", &run_on_analytic_params_kh, "Obtain Stokes I image with random B-field",
              py::arg("redshift"), py::arg("los_angle_deg"), py::arg("R_1_pc"),
              py::arg("b_0"),
              py::arg("m_b"),
              py::arg("K_1"), py::arg("n"),
              py::arg("s"), py::arg("gamma_min"), py::arg("background_fraction"),
              py::arg("Gamma_0"), py::arg("Gamma_1"), py::arg("betac_phi"),
              py::arg("number_of_pixels_along"), py::arg("number_of_pixels_across"),
              py::arg("lg_pixel_size_mas_start"), py::arg("lg_pixel_size_mas_stop"),
              py::arg("spiral_width_frac"),
              py::arg("phase_0"), py::arg("lambda_0"), py::arg("amp_0"),
              py::arg("phase_1"), py::arg("lambda_1"), py::arg("amp_1"),
              py::arg("phase_2"), py::arg("lambda_2"), py::arg("amp_2"),
              py::arg("t_obs_days"),
              py::arg("jet_side"));

}


