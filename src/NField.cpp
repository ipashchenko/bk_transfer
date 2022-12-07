#include "utils.h"
#include "NField.h"
#include "MyExceptions.h"


NField::NField(bool in_plasma_frame, ParticlesDistribution* particles, Geometry* geometry_out, Geometry* geometry_in,
               VField* vfield) :
    in_plasma_frame_(in_plasma_frame) {
    particles_ = particles;
    geometry_in_ = geometry_in;
    geometry_out_ = geometry_out;
    vfield_ = vfield;
}

double NField::nf(const Vector3d &point, const double t) const {
    double x, y, r_point, r_border;
    x = point[0];
    y = point[1];
    r_point = sqrt(x*x + y*y);
    double factor = 1.0;

    if(geometry_out_) {
        r_border = geometry_out_->radius_at_given_distance(point);
        if (r_point > r_border) {
            factor = exp(-pow(r_point - r_border, 2.0)/l_eps_N/l_eps_N);
        }
    }
    if(geometry_in_) {
        r_border = geometry_in_->radius_at_given_distance(point);
        if (r_point < r_border) {
            factor = exp(-pow(r_point - r_border, 2.0)/l_eps_N/l_eps_N);
        }
    }
    return _nf(point, t)*factor;
}

double NField::nf_plasma_frame(const Vector3d &point, const double gamma, const double t) const {
//    if (is_equipartition_) {
//        double n_nt_Bsq = particles_->get_equipartition_bsq_coefficient(b_sq);
//        return n_nt_Bsq*b_sq;
//    }
//    else {
//        double n = nf(point, t);
//        if (in_plasma_frame_) {
//            return n;
//        } else {
//            return n / gamma;
//        }
//    }
    double n = nf(point, t);
    if (in_plasma_frame_) {
        return n;
    } else {
        return n / gamma;
    }
}

NField::NField(NField *pField) {
    in_plasma_frame_ = pField->in_plasma_frame_;
    geometry_in_ = pField->geometry_in_;
    geometry_out_ = pField->geometry_out_;
    vfield_ = pField->vfield_;
    particles_ = pField->particles_;
}


BKNField::BKNField(double n_0, double n_n, ParticlesDistribution* particles, bool in_plasma_frame,
                   Geometry* geometry_out, Geometry* geometry_in, VField* vfield) :
        NField(in_plasma_frame, particles, geometry_out, geometry_in, vfield),
        n_0_(n_0),
        n_n_(n_n),
        amp_border_(1.0),
        r_mean_(0.0),
        r_width_(0.0),
        amp_axis_(1.0),
        r_width_axis_(0.0),
        background_fraction_(0.0),
        is_profile_set_(false),
        phases_0_(),
        lambdas_0_(),
        amps_0_(),
        k_(0.5),
        spiral_width_frac_(0.05),
        is_spirals_present_(false) {}


double BKNField::_nf(const Vector3d &point, double t) const {
    double r = point.norm();
    double raw_density = n_0_ * pow(r / pc, -n_n_);
    bool in_spiral = false;
    double spiral_sum = 0;

    double R_out = geometry_out_->radius_at_given_distance(point);


    // If we use heating models that depends on radius only
    if(is_profile_set_){
        double x = point[0];
        double y = point[1];
        double R_cur = hypot(x, y);
//        double R_out = geometry_out_->radius_at_given_distance(point);
        return (amp_axis_*generalized1_gaussian1d(R_cur / R_out, 0.0, r_width_axis_, 2) + amp_border_*generalized1_gaussian1d(R_cur / R_out, r_mean_, r_width_, 2) + background_fraction_) * raw_density;
    }
    // If we are modelling KH modes
    else if (is_spirals_present_){
        double x = point[0];
        double y = point[1];
        double z = point[2];
        double R_cur = hypot(x, y);
        for(int i=0;i<number_of_spirals();i++){
            double x_sp = amps_0_[i] * pow(abs(z)/pc, k_) * sin(2*M_PI*abs(z)/(lambdas_0_[i]*pow(abs(z)/pc, k_)) + phases_0_[i]);
            double y_sp = amps_0_[i] * pow(abs(z)/pc, k_) * cos(2*M_PI*abs(z)/(lambdas_0_[i]*pow(abs(z)/pc, k_)) + phases_0_[i]);
            double distance = hypot(x-x_sp, y-y_sp);
            if(distance < spiral_width_frac_*R_cur){
                in_spiral = true;
            }
            spiral_sum += raw_density*generalized1_gaussian1d(distance, 0.0, spiral_width_frac_*R_out, 2);
        }
        return spiral_sum + raw_density*background_fraction_;
//        if(in_spiral){
//            return raw_density;
//        }
//        else{
//            return raw_density*background_fraction_;
//        }
    }
    // If density is constant at given z
    else {
        return raw_density;
    }
}


void BKNField::set_heating_profile(double amp_border, double r_mean, double r_width, double amp_axis, double r_width_axis, double background_fraction) {
    if (r_mean > 1 || r_mean < 0.0 || r_width < 0.0 || background_fraction < 0.0 || background_fraction_ > 1.0){
        throw BadHeatingProfileParameters();
    }
    amp_border_ = amp_border;
    r_mean_ = r_mean;
    r_width_ = r_width;
    amp_axis_ = amp_axis;
    r_width_axis_ = r_width_axis;
    background_fraction_ = background_fraction;
    is_profile_set_ = true;
}


void BKNField::set_background_fraction(double background_fraction) {
    background_fraction_ = background_fraction;
}

void BKNField::set_spiral(double phase_0, double lambda_0, double amp_0) {
    phases_0_.push_back(phase_0);
    lambdas_0_.push_back(lambda_0);
    amps_0_.push_back(amp_0);
    is_spirals_present_ = true;
}



EquipartitionBKNfield::EquipartitionBKNfield(ParticlesDistribution *particles, std::vector<ScalarBField*> bfields,
                                             Geometry *geometry_out, Geometry *geometry_in, VField *vfield,
                                             double t_start, double fraction):
        NField(true, particles, geometry_out, geometry_in, vfield),
        t_start_(t_start),
        fraction_(fraction) {
    bfields_ = bfields;
}

double EquipartitionBKNfield::_nf(const Vector3d &point, double t) const {
    auto v = vfield_->vf(point);
    double b = 0.0;
    for(auto bfield: bfields_) {
        b += bfield->bf_plasma_frame(point, v, t);
    }
    double raw_density = fraction_*particles_->get_equipartition_bsq_coefficient()*b*b;
    return raw_density;
}


EquipartitionKHNfield::EquipartitionKHNfield(ParticlesDistribution *particles, BKScalarBField *bfield,
                                             Geometry *geometry_out, Geometry *geometry_in, VField *vfield,
                                             double omega, double t_start):
        NField(true, particles, geometry_out, geometry_in, vfield),
        background_fraction_(0.0),
        phases_0_(),
        lambdas_0_(),
        amps_0_(),
        k_(0.5),
        scale_({1.0, 1.0, 1.0}),
        t_start_(t_start),
        omega_(omega),
        spiral_width_frac_() {
    bfield_ = bfield;
}

double EquipartitionKHNfield::_nf(const Vector3d &point, double t) const {
    auto v = vfield_->vf(point);
    double b = bfield_->bf_plasma_frame(point, v, t);
//    std::cout << "B = " << b << "\n";
    double raw_density = particles_->get_equipartition_bsq_coefficient()*b*b;
//    std::cout << "N = " << raw_density << "\n";
    double spiral_sum = 0;
    double R_out = geometry_out_->radius_at_given_distance(point);

    // If we are modelling KH modes
    double x = point[0];
    double y = point[1];
    double z = point[2];
    double R_cur = hypot(x, y);
    for(int i=0;i<number_of_spirals();i++){
        double x_sp = amps_0_[i] * pow(abs(z)/pc, k_) * cos(2*M_PI*abs(z)/(lambdas_0_[i]*pow(abs(z)/pc, k_)) - omega_*t + phases_0_[i]);
        double y_sp = -amps_0_[i] * pow(abs(z)/pc, k_) * sin(2*M_PI*abs(z)/(lambdas_0_[i]*pow(abs(z)/pc, k_)) - omega_*t + phases_0_[i]);
        double distance = hypot(x-x_sp, y-y_sp);
        spiral_sum += scale_[i]*raw_density*generalized1_gaussian1d(distance, 0.0, spiral_width_frac_[i]*R_out, 8.0);
    }
    return spiral_sum + raw_density*background_fraction_;
}

void EquipartitionKHNfield::set_background_fraction(double background_fraction) {
    background_fraction_ = background_fraction;
}

void EquipartitionKHNfield::set_spiral(double phase_0, double lambda_0, double amp_0) {
    phases_0_.push_back(phase_0);
    lambdas_0_.push_back(lambda_0);
    amps_0_.push_back(amp_0);
}

void EquipartitionKHNfield::set_spiral_width_frac(std::vector<double> frac) {
    spiral_width_frac_ = frac;
}

void EquipartitionKHNfield::set_spiral_scale(std::vector<double> scale) {
    scale_ = scale;
}

FlareBKNField::FlareBKNField(NField* bkg_nfield, double amp, double t_start, double width_pc, VField* flare_pattern_vfield) :
        NField(bkg_nfield),
        amp_(amp),
        t_start_(t_start),
        width_pc_(width_pc)
        {
        bkg_nfield_ = bkg_nfield;
        flare_pattern_vfield_ = flare_pattern_vfield;
        }

//// TODO: Implement equipartition!!!
//// Note: original implementation of LTT delay through the apparent velocity. Only applicable for straight jets.
//double FlareBKNField::_nf(const Vector3d &point, double t) const {
//    // Direction to the observer
//	double theta_los_ = 5.74*M_PI/180.;
//    Vector3d n_los = {sin(theta_los_), 0, cos(theta_los_)};
//    Vector3d v = flare_pattern_vfield_->vf(point);
//    Vector3d v_hat = v.normalized();
//    double cos_theta_local = v_hat.dot(n_los);
////	std::cout << "cos_theta = " << cos_theta_local << "\n";
//    double beta = v.norm()/c;
////	std::cout << "beta = " << beta << "\n";
//	double z_ = 0;
//    double beta_app = beta/(1.0 - beta*cos_theta_local)/(1.0 + z_);
////	std::cout << "beta_app = " << beta_app << "\n";
//    double r = point.norm();
////	return n_0_ * pow(r/pc, -n_n_) * exp(-pow(r - beta_app*c*(t - t_start_), 2.0)/(width_pc_*width_pc_*pc*pc));
////	double result = 500000 * pow(r/pc, -2) * exp(-pow(r - beta_app*c*(t - t_start_), 2.0)/(width_pc_*width_pc_*pc*pc));
////	double result = amp_ * bkg_nfield_->_nf(point, t) * exp(-pow(r - beta_app*c*(t - t_start_), 2.0)/(width_pc_*width_pc_*pc*pc));
//	// FIXME: This works
//	double result = amp_ * bkg_nfield_->_nf(point, t) * generalized1_gaussian1d(r, beta_app*c*(t - t_start_), width_pc_*pc, 20.0);
//
////	return amp_ * bkg_nfield_->_nf(point, t) * generalized1_gaussian1d(r, beta_app*c*(t - t_start_), width_pc_*pc, 20.0);
////	double result = amp_ * bkg_nfield_->_nf(point, t) * generalized1_gaussian1d(r, beta_app*c*(t - t_start_), width_pc_*pc, 20.0);
////	if(result > 0) {
////		std::cout << "Flare N = " << result << "\n";
////	}
//	return result;
//}


// FIXME: Account redshift time (1 + z)!!!
// Slow light approximation!!!
double FlareBKNField::_nf(const Vector3d &point, double t) const {
	// TODO: Here we should integrate flare pattern speed in time to find the position of the flare (e.g. blob center).
	// We can model flare as a spherical blob.
    Vector3d v = flare_pattern_vfield_->vf(point);
    double r = point.norm();
//	return amp_ * bkg_nfield_->_nf(point, t) * exp(-pow(r - v.norm()*(t - t_start_), 2.0)/(width_pc_*width_pc_*pc*pc));
	return amp_ * bkg_nfield_->_nf(point, t) * generalized1_gaussian1d(r, v.norm()*(t - t_start_), width_pc_*pc, 20.0);
}
