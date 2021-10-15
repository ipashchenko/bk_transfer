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
    double n = nf(point, t);
    if (in_plasma_frame_) {
        return n;
    } else {
        return n/gamma;
    }
}


BKNField::BKNField(double n_0, double n_n, ParticlesDistribution* particles, bool in_plasma_frame,
                   Geometry* geometry_out, Geometry* geometry_in, VField* vfield) :
        NField(in_plasma_frame, particles, geometry_out, geometry_in, vfield),
        n_0_(n_0),
        n_n_(n_n),
        r_mean_(0.0),
        r_width_(0.0),
        background_fraction_(0.025),
        is_profile_set_(false),
        phases_0_(),
        lambdas_0_(),
        amps_0_(),
        k_(0.5),
        spiral_width_frac_(0.1),
        is_spirals_present_(false) {}


double BKNField::_nf(const Vector3d &point, const double t) const {
    double r = point.norm();
    double raw_density = n_0_ * pow(r / pc, -n_n_);
    bool in_spiral = false;

    // If we use heating models that depends on radius only
    if(is_profile_set_){
        double x = point[0];
        double y = point[1];
        double R_cur = hypot(x, y);
        double R_out = geometry_out_->radius_at_given_distance(point);
        return (generalized1_gaussian1d(R_cur / R_out, r_mean_, r_width_, 2) + background_fraction_) * raw_density;
    }
    // If we are modelling KH modes
    else if (is_spirals_present_){
        double x = point[0];
        double y = point[1];
        double z = point[2];
        double R_cur = hypot(x, y);
        for(int i=0;i<number_of_spirals();i++){
            double x_sp = amps_0_[i] * pow(abs(z)/pc, k_) * sin(2*M_PI*abs(z)/(lambdas_0_[i]*pow(abs(z)/pc, k_)) + phases_0_[i]);
            double y_sp = amps_0_[i] * pow(abs(z)/pc, k_) * cos(2*M_PI*abs(z)/(lambdas_0_[0]*pow(abs(z)/pc, k_)) + phases_0_[i]);
            double distance = hypot(x-x_sp, y-y_sp);
            if(distance < spiral_width_frac_*R_cur){
                in_spiral = true;
            }
        }
        if(in_spiral){
            return raw_density;
        }
        else{
            return raw_density*background_fraction_;
        }
    }
    // If density is constant at given z
    else {
        return raw_density;
    }
}


void BKNField::set_heating_profile(double r_mean, double r_width, double background_fraction) {
    if (r_mean > 1 || r_mean < 0.0 || r_width < 0.0 || background_fraction < 0.0 || background_fraction_ > 1.0){
        throw BadHeatingProfileParameters();
    }
    r_mean_ = r_mean;
    r_width_ = r_width;
    background_fraction_ = background_fraction;
    is_profile_set_ = true;
}


void BKNField::set_spiral(double phase_0, double lambda_0, double amp_0) {
    phases_0_.push_back(phase_0);
    lambdas_0_.push_back(lambda_0);
    amps_0_.push_back(amp_0);
    is_spirals_present_ = true;
}


FlareBKNField::FlareBKNField(double n_0, double n_n, double t_start, double width_pc, ParticlesDistribution* particles,
                             bool in_plasma_frame, double theta_los, double z,
                             Geometry* geometry_out, Geometry* geometry_in, VField* vfield) :
        NField(in_plasma_frame, particles, geometry_out, geometry_in, vfield),
        n_0_(n_0),
        n_n_(n_n),
        t_start_(t_start),
        width_pc_(width_pc),
        z_(z),
        theta_los_(theta_los)
        {}

double FlareBKNField::_nf(const Vector3d &point, const double t) const {
    // Direction to the observer
    Vector3d n_los = {sin(theta_los_), 0, cos(theta_los_)};
//    Vector3d v = vfield_->vf(point).normalized();
    Vector3d v = vfield_->vf(point);
    Vector3d v_hat = v.normalized();
    double cos_theta_local = v_hat.dot(n_los);
    double sin_theta_local = sqrt(1.0 - cos_theta_local*cos_theta_local);
    double beta = v.norm()/c;
    double beta_app = beta/(1.0 - beta*cos_theta_local)/(1.0 + z_);

//    std::cout << "beta_app = " << beta_app << "\n";

    double r = point.norm();
    return n_0_ * pow(r/pc, -n_n_) * exp(-pow(r*sin_theta_local - beta_app*c*(t - t_start_), 2.0)/(width_pc_*width_pc_*pc*pc));
}