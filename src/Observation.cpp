#include <string>
#include <iostream>
#include "Observation.h"
#include "System.h"
#include <boost/range/algorithm.hpp>
#include <boost/numeric/odeint.hpp>
#include <utility>

using namespace boost::numeric::odeint;


Observation::Observation(Jet *newjet, ImagePlane *newimagePlane)
{
  jet = newjet;
  imagePlane = newimagePlane;
};

// ``dt_max`` - max. step size in pc. Adaptive step size will be kept less then this.
// ``n`` defines the initial step size (that will be adjusted) through utils.steps_schedule function.
void Observation::observe(int n, double tau_max, double dt_max, double tau_min, double nu, string polarization,
                          double relerr, double t_obs) {

    jet->set_t_obs(t_obs);
	dt_max *= pc;
    auto image_size = getImageSize();
	vector<Pixel>& pixels = imagePlane->getPixels();
	vector<Ray>& rays = imagePlane->getRays();
	omp_set_num_threads(4);



    // FIXME: Debug
//    unsigned long int j = 63;
//    unsigned long int k = 2;
//    auto &ray = rays[j*image_size.second+k];
//    auto &pxl = pixels[j*image_size.second+k];
//    observe_single_pixel(ray, pxl, tau_min, tau_max, n, dt_max, nu, polarization, relerr);



//	 Comment out for easy debug printing
	#pragma omp parallel for schedule(dynamic) collapse(2) default(none) shared(image_size, rays, pixels, tau_min, tau_max, n, dt_max, nu, polarization, relerr)
    for (unsigned long int j = 0; j < image_size.first; ++j) {
	    // TODO: If one (doesn't?) need counter-jet side -- start from uncommenting this line:D
	    // TODO: I thik cj is now handled differently!
//      for (unsigned long int k = image_size.second/2-50; k < image_size.second; ++k) {
        for (unsigned long int k = 0; k < image_size.second; ++k) {
            auto &ray = rays[j*image_size.second+k];
        	auto &pxl = pixels[j*image_size.second+k];

//        	std::cout << "Observing pixel " << j << ", " << k << std::endl;
        	observe_single_pixel(ray, pxl, tau_min, tau_max, n, dt_max, nu, polarization, relerr);
        }
    }


}


vector<vector<double>> Observation::getImage(string value) {
  return imagePlane->getImage(std::move(value));
}

pair<unsigned long int, unsigned long int> Observation::getImageSize() {
  return imagePlane->image_size;
}

pair<double, double> Observation::integrate_tau_adaptive(std::list<Intersection> &list_intersect, Vector3d ray_direction,
                                                         const double nu, double tau_max, int n, double dt_max,
                                                         double relerr) {
//    std::cout << "Begin tau integration!" << std::endl;
    // Precision for finding tau crossing
	const double tau_precision = 0.01;

	// Write final values in this variables inside for-cycle
	double background_tau = 0.;
	double thickness = 0.;

	for (auto it = list_intersect.begin(); it != list_intersect.end(); ++it) {
		auto borders = (*it).get_path();

		Vector3d point_in = borders.first;
		Vector3d point_out = borders.second;

//		std::cout << "In = " << point_in/pc << ", Out = " << point_out/pc << "\n";

		double length = (point_out - point_in).norm();
		double dt = length/n;

		// Directed to observer (as we integrate ``k`` only)
		Vector3d ray_direction_ = -1. * ray_direction;

		// First integrate from the closest to the observer point till some ``tau_max``
		// Here ``ray_direction`` points to the observer as it is an argument of k_I
		Tau tau(jet, point_in, ray_direction_, nu);
		// This is out State
		double optDepth = 0.0;
		typedef runge_kutta_dopri5< double > stepper_type;
	    auto stepper = make_dense_output(abserr, relerr, dt_max, stepper_type());
		auto is_done = std::bind(check_opt_depth, tau_max, std::placeholders::_1);
		auto ode_range = make_adaptive_range(std::ref(stepper), tau, optDepth, 0.0, length, dt);
		auto found_iter = std::find_if(ode_range.first, ode_range.second, is_done);

		// ``tau_max`` somewhere on interval considered
		if (found_iter != ode_range.second) {
		    double found_tau = found_iter.get_state();
			// The dense out stepper now covers the interval where the condition changes
            // Improve the solution by bisection
            double t0 = stepper.previous_time();
            double t1 = stepper.current_time();
            double t_m;
            double x_m = found_tau;

            // Use odeint's resizing functionality to allocate memory for x_m
            // Adjust_size_by_resizeability(x_m, optDepth, typename is_resizeable<double>::type());
            while(std::abs(x_m - tau_max) > tau_precision) {
                std::cout << "Finding cross tau!" << std::endl;
                // Get the mid point time
                t_m = 0.5 * (t0 + t1);
                // Obtain the corresponding state
                stepper.calc_state(t_m, x_m);
                if (x_m > tau_max)
                    // Condition changer lies before midpoint
                    t1 = t_m;
                else
                    // Condition changer lies after midpoint
                    t0 = t_m;
            }
            // We found the interval of size eps, take it's midpoint as final guess
            t_m = 0.5 * (t0 + t1);
            stepper.calc_state(t_m, x_m);
            std::cout << "Found precise cross value tau = " << x_m << std::endl;
            std::cout << "Originally it was tau = " << found_iter.get_state() << std::endl;
            double t_tau_max = t_m;
            // double t_tau_max = stepper.current_time();

            Ray ray_tau_max(point_in, ray_direction);
            Vector3d point_out_tau_max = ray_tau_max.point(t_tau_max);
            // Set point of tau=tau_max as outer border for further integrations
            it.operator*().set_point_out(point_out_tau_max);

            // Delete all other furthest intersections if any
            if (list_intersect.size() > 1) {
                ++it;
                // This now is container.end()
                it = list_intersect.erase(it, list_intersect.end());
                // Move it before container.end() to make it end() on the new cycle
                --it;
            }
		}
        // Update background value (or write final values if this is last cycle)
        background_tau += found_iter.get_state();
        thickness += length;
	}
//	std::cout << "L[pc] = " << thickness/pc << "\n";
//    std::cout << "tau = " << background_tau << "\n";
//    std::cout << "Done tau integration!" << std::endl;
	return std::make_pair(background_tau, thickness);
}

void Observation::integrate_i_adaptive(std::list<Intersection> &list_intersect, const Vector3d& ray_direction, const double nu,
                                       int n, double& background_I, double dt_max, double relerr) {
//    std::cout << "Doing adaprive I integration..." << "\n";

    for (auto it = list_intersect.rbegin();
         it != list_intersect.rend(); ++it) {
        auto borders = (*it).get_path();
        Vector3d point_in = borders.first;
        Vector3d point_out = borders.second;

        double length = (point_out - point_in).norm();
        // Initial step size (will be adjusted)
        double dt = length / n;

        Vector3d inv_direction = -1. * ray_direction;
        // Here ``inv_direction`` points to the observer as it is an argument of k_I
        I stokesI(jet, point_out, inv_direction, nu);

        double stI = background_I;

        // Adaptive
        typedef runge_kutta_dopri5<double> stepper_type;
//         One can add observer function at the end of the argument list. Here ``dt`` is the initial step size
        int num_steps = integrate_adaptive(make_controlled(abserr, relerr, dt_max, stepper_type()),
                                           stokesI,
                                           stI, 0.0, length, dt);
//        std::cout << "Num.steps for I : " << num_steps << "\n";

        // Const step size
//        typedef runge_kutta4<double> stepper_type;
//        integrate_const(stepper_type(), stokesI, stI, 0.0, length, dt);

        background_I = stI;
    }

//    std::cout << "Done I integration!" << std::endl;

}

void Observation::integrate_speed_adaptive(std::list<Intersection> &list_intersect, const Vector3d& ray_direction, const double nu,
                                           int n, double& emm_weighted_beta_app, double dt_max, double relerr) {

    for (auto it = list_intersect.rbegin();
         it != list_intersect.rend(); ++it) {
        auto borders = (*it).get_path();
        Vector3d point_in = borders.first;
        Vector3d point_out = borders.second;

        double length = (point_out - point_in).norm();
        // Initial step size (will be adjusted)
        double dt = length / n;

        Vector3d inv_direction = -1. * ray_direction;
        Speed speed(jet, point_out, inv_direction, nu);

        double beta_app = emm_weighted_beta_app;

        // Adaptive
        typedef runge_kutta_dopri5<double> stepper_type;
        // One can add observer function at the end of the argument list. Here ``dt`` is the initial step size
        int num_steps = integrate_adaptive(make_controlled(abserr, relerr, dt_max, stepper_type()),
                                           speed,
                                           beta_app, 0.0, length, dt);

        // Const step size
//        typedef runge_kutta4<double> stepper_type;
//        integrate_const(stepper_type(), speed, beta_app, 0.0, length, dt);

        emm_weighted_beta_app = beta_app;
    }
}

void Observation::integrate_full_stokes(std::list<Intersection> &list_intersect, const Vector3d& ray_direction,
                                        const double nu, int n, std::vector<double>& background, double dt_max) {
    for (auto it = list_intersect.rbegin(); it != list_intersect.rend(); ++it) {
        auto borders = (*it).get_path();
        Vector3d point_in = borders.first;
        Vector3d point_out = borders.second;
        double length = (point_out - point_in).norm();
        double dt = length / n;

        Vector3d inv_direction = -1. * ray_direction;
        FullStokes full_stokes(jet, point_out, inv_direction, nu);
        typedef runge_kutta4<std::vector<double>> stepper_type;
        auto stepper = stepper_type();
        std::vector<double> iquv = background;
        integrate_const(stepper, full_stokes, iquv, 0.0, length, dt);
        background = iquv;
    }
}

void Observation::integrate_full_stokes_adaptive(std::list<Intersection> &list_intersect, const Vector3d& ray_direction,
                                                 const double nu, int n, std::vector<double>& background, double dt_max,
                                                 double relerr) {

	for (auto it = list_intersect.rbegin();
	     it != list_intersect.rend(); ++it) {
		auto borders = (*it).get_path();
		Vector3d point_in = borders.first;
		Vector3d point_out = borders.second;

		double length = (point_out - point_in).norm();
		double dt = length / n;

		Vector3d inv_direction = -1. * ray_direction;
		FullStokes full_stokes(jet, point_out, inv_direction, nu);
		typedef runge_kutta_dopri5<std::vector<double>> stepper_type;
		auto stepper = stepper_type();

        std::vector<double> iquv = background;
		// One can add observer function at the end of the argument list.
		// TODO: Previously it was 1E-16, 1E-16
		int num_steps = integrate_adaptive(make_controlled(abserr, relerr, dt_max, stepper_type()),
		                   full_stokes, iquv, 0.0, length, dt);
		background = iquv;
	}
}


void Observation::integrate_faraday_rotation_depth_adaptive(std::list<Intersection> &list_intersect, const Vector3d& ray_direction,
                                                 const double nu, int n, double& background, double dt_max, double relerr) {

    for (auto it = list_intersect.rbegin();
         it != list_intersect.rend(); ++it) {
        auto borders = (*it).get_path();
        Vector3d point_in = borders.first;
        Vector3d point_out = borders.second;

        double length = (point_out - point_in).norm();
        double dt = length / n;

        //std::cout << "dt = " << dt << std::endl;

        Vector3d inv_direction = -1. * ray_direction;
        TauFR tau_fr(jet, point_out, inv_direction, nu);
        typedef runge_kutta_dopri5<double> stepper_type;
        auto stepper = stepper_type();

        // State
        double tau_fr_value = background;
        // One can add observer function at the end of the argument list.
        // TODO: Previously it was 1E-16, 1E-16
        int num_steps = integrate_adaptive(make_controlled(abserr, relerr, dt_max, stepper_type()),
                                           tau_fr, tau_fr_value, 0.0, length, dt);
        //std::cout << "num_steps = " << num_steps << std::endl;
        //std::cout << "tau_fr = " << tau_fr_value << std::endl;

        background = tau_fr_value;
    }
}

void Observation::observe_single_pixel(Ray &ray, Pixel &pxl,  double tau_min, double tau_max, int n, double dt_max,
                                       double nu, string polarization, double relerr) {
    auto ij = pxl.getij();
//    std::cout << "=====================================================================================================" << "\n";
//    std::cout << "Observing pixel " << ij.first << ", " << ij.second << "\n";
//    std::cout << "Coordinate = " << pxl.getCoordinate()/pc << "\n";
//    std::cout << "=====================================================================================================" << "\n";
    auto ray_direction = ray.direction();
    std::list<Intersection> list_intersect = jet->hit(ray);
    if (!list_intersect.empty()) {
        std::pair<double, double> tau_l_end;
        tau_l_end = integrate_tau_adaptive(list_intersect, ray_direction, nu, tau_max, n, dt_max, relerr);
//        std::cout << "Tau is done!" << "\n";
        double background_tau = tau_l_end.first;
        double thickness = tau_l_end.second;

        // Write final values to this inside function integrate_...
        double background_I = 0.;
        double emm_weighted_beta_app = 0.;
        double background_taufr = 0.;
        std::vector<double> background_iquv{0., 0., 0., 0};

        // Calculate I only if optical depth is high enough (> tau_min)
        if (background_tau > tau_min && background_tau < 10*tau_max) {
            if (polarization == "I") {
                integrate_i_adaptive(list_intersect, ray_direction, nu, n, background_I, dt_max, relerr);
            }
            else if(polarization == "speed") {
                integrate_i_adaptive(list_intersect, ray_direction, nu, n, background_I, dt_max, relerr);
                integrate_speed_adaptive(list_intersect, ray_direction, nu, n, emm_weighted_beta_app, dt_max, relerr);
            }
            else if (polarization == "full") {
                // TODO: If one does not need adaptive step size - uncomment this line
                //integrate_full_stokes(list_intersect, ray_direction, nu, n, background_iquv, dt_max);
                integrate_full_stokes_adaptive(list_intersect, ray_direction, nu, n, background_iquv, dt_max, relerr);
                integrate_faraday_rotation_depth_adaptive(list_intersect, ray_direction, nu, n, background_taufr,
                                                          dt_max, relerr);
            }
        }
        else{
            std::cout << "Tau = " << background_tau << "\n";
        }
//        std::cout << "I is done!" << "\n";

        // Write values to current pixel
        std::string value("tau");
        pxl.setValue(value, background_tau);

        if (polarization == "I") {
            value = "I";
            pxl.setValue(value, background_I);
        }
        else if (polarization == "speed") {
            value = "I";
            pxl.setValue(value, background_I);
            value = "SPEED";
            pxl.setValue(value, emm_weighted_beta_app);
        }
        else if (polarization == "full") {
            value = "I";
            pxl.setValue(value, background_iquv[0]);
            value = "Q";
            pxl.setValue(value, background_iquv[1]);
            value = "U";
            pxl.setValue(value, background_iquv[2]);
            value = "V";
            pxl.setValue(value, background_iquv[3]);
            value = "tau_fr";
            pxl.setValue(value, background_taufr);
        }
        value = "l";
        pxl.setValue(value, thickness);
    }
}
