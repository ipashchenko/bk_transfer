#include <cmath>
#include <Eigen/Eigen>
#include <utility>
#include <utils.h>
#include "VField.h"
#include "MyExceptions.h"

using Eigen::Vector3d;


VField::VField(Geometry* geometry): r1_(0.0), r2_(0.0), K1_(1.0), K2_(1.0), b1_(1.0), b2_(1.0), b3_(1.0), is_profile_set(false) {
    geometry_ = geometry;
}

void VField::set_profile(double r1, double r2, double K1, double K2, double b1, double b2, double b3) {
    r1_ = r1;
    r2_ = r2;
    K1_ = K1;
    K2_ = K2;
    b1_ = b1;
    b2_ = b2;
    b3_ = b3;
    is_profile_set = true;
}


ConstFlatVField::ConstFlatVField(double gamma, Geometry* geometry, double betac_phi) :
    VField(geometry), gamma_(gamma), betac_phi_(betac_phi) {};

Vector3d ConstFlatVField::vf(const Vector3d &point) const {
    double x = point[0];
    double y = point[1];
    double z = point[2];
    double phi = atan2(y, x);
    // As atan2 returns [-pi, pi], put this to [0, 2pi]
    if(phi < 0){
        phi += 2.0*M_PI;
    }

    double v_phi = c*betac_phi_;
    double r_border = geometry_->radius_at_given_distance(point);
    double r_point = sqrt(x*x + y*y);

    double gamma = gamma_;
    if(is_profile_set) {
        gamma *= density_profile(r_point/r_border, r1_, r2_, K1_, K2_, b1_, b2_, b3_);
    }
    double v_pol = c*sqrt(gamma*gamma - 1.0)/gamma;


    Vector3d V_p;
    if(point[2] > 0){
        V_p = {0.0, 0.0, v_pol};
    } else {
        V_p = {0.0, 0.0, -v_pol};
    }
    // FIXME: Move in if(z > 0) to account for the different direction of the rotation?
    Vector3d V_phi = {-sin(phi)*v_phi, cos(phi)*v_phi, 0.0};

    return V_p + V_phi;

//    if(z > 0) {
//        // Relativistic addition of velocities Gamma (along jet) and phi-component (perpendicular jet)
//        //return {-sin(phi)*v_phi/gamma, cos(phi)*v_phi/gamma, c*sqrt(1. - 1./(gamma*gamma))};
//        return {sin(phi)*v_phi/gamma, -cos(phi)*v_phi/gamma, c*sqrt(1. - 1./(gamma*gamma))};
//    } else {
//        return {sin(phi)*v_phi/gamma, -cos(phi)*v_phi/gamma, -c*sqrt(1. - 1./(gamma*gamma))};
//    }
}


ConstCentralVField::ConstCentralVField(double gamma, Geometry* geometry, double betac_phi, Vector3d origin) :
    VField(geometry), gamma_(gamma), betac_phi_(betac_phi), origin_(std::move(origin)) {}

Vector3d ConstCentralVField::vf(const Vector3d &point) const {
    double x = point[0];
    double y = point[1];
    double z = abs(point[2]);
    double phi = atan2(y, x);
    // As atan2 returns [-pi, pi], put this to [0, 2pi]
    if(phi < 0){
        phi += 2.0*M_PI;
    }
    double v_phi = c*betac_phi_;
    double r_border = geometry_->radius_at_given_distance(point);
    double r_cur = hypot(x, y);

    // Poloidal angle - always from the (0,0,0) to the current point (central velocity field)
    double tan_polangle = r_cur/z;
    double sinz = sqrt(tan_polangle*tan_polangle/(1.0 + tan_polangle*tan_polangle));
    double cosz = sqrt(1.0/(1.0 + tan_polangle*tan_polangle));


    double gamma = gamma_;
    if(is_profile_set) {
        gamma *= density_profile(r_cur/r_border, r1_, r2_, K1_, K2_, b1_, b2_, b3_);
    }

    double v_pol = c*sqrt(1. - 1./(gamma*gamma));
//    if(z > 0) {
//        return {v_r*(x-origin_[0])/r - sin(phi)*v_phi, v_r*(y-origin_[1])/r + cos(phi)*v_phi, v_r*(z-origin_[2])/r};
//    } else {
//        // FIXME: z < 0
//        return {v_r*(x-origin_[0])/r + sin(phi)*v_phi, v_r*(y-origin_[1])/r - cos(phi)*v_phi, v_r*(abs(z)-origin_[2])/r};
//    }
    Vector3d V_p;
    if(point[2] > 0){
        V_p = {v_pol*sinz*cos(phi), v_pol*sinz*sin(phi), v_pol*cosz};
    } else {
        V_p = {v_pol*sinz*cos(phi), v_pol*sinz*sin(phi), -v_pol*cosz};
    }

    Vector3d V_phi = {-sin(phi)*v_phi, cos(phi)*v_phi, 0.0};

    return V_p + V_phi;
};


//ShearedFlatVField::ShearedFlatVField(double gamma_axis, double gamma_border, double r) :
//    gamma_axis_(gamma_axis), gamma_border_(gamma_border), r_(r) {}
//
//Vector3d ShearedFlatVField::vf(const Vector3d &point) const {
//    double x = point[0];
//    double y = point[1];
//    double z = point[2];
//    double r = sqrt(x*x + y*y);
//    double gamma = gamma_axis_-(gamma_axis_-gamma_border_)*r/r_;
//    if(z > 0) {
//        return {0, 0, c*sqrt(1. - 1./(gamma*gamma))};
//    } else {
//        return {0, 0, -c*sqrt(1. - 1./(gamma*gamma))};
//    }
//}

SheathFlatVField::SheathFlatVField(double gamma_spine_0, double gamma_sheath_0,
                                   Geometry* geometry_in, Geometry* geometry_out,
                                   double gamma_spine_v, double gamma_sheath_v,
                                   double betac_phi_in,  double betac_phi_out) :
    VField(geometry_out),
    gamma_spine_0_(gamma_spine_0),
    gamma_spine_v_(gamma_spine_v),
    gamma_sheath_0_(gamma_sheath_0),
    gamma_sheath_v_(gamma_sheath_v),
    geometry_in_(geometry_in),
    betac_phi_in_(betac_phi_in),
    betac_phi_out_(betac_phi_out)  {}

Vector3d SheathFlatVField::vf(const Vector3d &point) const {
    double x = point[0];
    double y = point[1];
    double z = point[2];
    double phi = atan2(y, x);
    double r_border_out = geometry_->radius_at_given_distance(point);
    double r_border_in = geometry_in_->radius_at_given_distance(point);
    double r_point = sqrt(x*x + y*y);

    double gamma;
    double v_phi;
    if (r_point < r_border_in) {
        gamma = gamma_spine_0_ + gamma_spine_v_*abs(z/pc);
        v_phi = c*betac_phi_in_;
    }
    else if (r_point < r_border_out){
        gamma = gamma_sheath_0_ + gamma_sheath_v_*abs(z/pc);
        v_phi = c*betac_phi_out_;
    } else {
        gamma = 1.0;
        v_phi = 0.0;
    }

    if(z > 0) {
        return {-sin(phi)*v_phi/gamma, cos(phi)*v_phi/gamma, c*sqrt(1. - 1./(gamma*gamma))};
        //return {sin(phi)*v_phi, -cos(phi)*v_phi, c*sqrt(1. - 1./(gamma*gamma))};
    } else {
        return {sin(phi)*v_phi/gamma, -cos(phi)*v_phi/gamma, -c*sqrt(1. - 1./(gamma*gamma))};
    }
}


//ShearedCentralVField::ShearedCentralVField(double gamma_axis, double gamma_border, double theta, Vector3d origin) :
//    gamma_axis_(gamma_axis), gamma_border_(gamma_border), theta_(theta), origin_(std::move(origin)) {}
//
//Vector3d ShearedCentralVField::vf(const Vector3d &point) const {
//    double x = point[0];
//    double y = point[1];
//    double z = point[2];
//    double r = (point - origin_).norm();
//    double theta = acos((z-origin_[2])/r);
//    double gamma = gamma_axis_+(gamma_border_-gamma_axis_)*theta/theta_;
//    double v_r = c*sqrt(1. - 1./(gamma*gamma));
//    return {v_r*(x-origin_[0])/r, v_r*(y-origin_[1])/r, v_r*(z-origin_[2])/r};
//}
//
//SheathCentralVField::SheathCentralVField(double gamma_spine, double gamma_sheath, double theta_sheath, Vector3d origin) :
//    gamma_spine_(gamma_spine), gamma_sheath_(gamma_sheath), theta_sheath_(theta_sheath), origin_(std::move(origin)) {}
//
//Vector3d SheathCentralVField::vf(const Vector3d &point) const {
//    double x = point[0];
//    double y = point[1];
//    double z = point[2];
//    double r = (point - origin_).norm();
//    double theta = acos((z-origin_[2])/r);
//    double gamma;
//    if (theta < theta_sheath_) {
//        gamma = gamma_spine_;
//    } else {
//        gamma = gamma_sheath_;
//    }
//    double v_r = c*sqrt(1. - 1./(gamma*gamma));
//    return {v_r*(x-origin_[0])/r, v_r*(y-origin_[1])/r, v_r*(z-origin_[2])/r};
//}


ConstParabolicVField::ConstParabolicVField(double gamma, Geometry* geometry, double betac_phi, Vector3d origin) :
        VField(geometry), gamma_(gamma), betac_phi_(betac_phi), origin_(std::move(origin)) {}

Vector3d ConstParabolicVField::vf(const Vector3d &point) const {
    double x = point[0];
    double y = point[1];
    double z = abs(point[2]);
    double r_cur = hypot(x, y);

    double phi = atan2(y, x);
    // As atan2 returns [-pi, pi], put this to [0, 2pi]
    if(phi < 0){
        phi += 2.0*M_PI;
    }

    double r_border = geometry_->radius_at_given_distance(point);
    double gamma = gamma_;
    if(is_profile_set) {
        gamma *= density_profile(r_cur/r_border, r1_, r2_, K1_, K2_, b1_, b2_, b3_);
    }

    double v_phi = c*betac_phi_;
    double v_pol = c*sqrt(gamma*gamma - 1.0)/gamma;

    // Constant in the parabolic equation for a given streamline
    double R_1_cur = r_cur/ sqrt(z);
    // Derivative of the local parabola equation at given point to find the tangent of the poloidal angle
    double tan_polangle = 0.5*R_1_cur/ sqrt(z);

    double sinz = sqrt(tan_polangle*tan_polangle/(1.0 + tan_polangle*tan_polangle));
    double cosz = sqrt(1.0/(1.0 + tan_polangle*tan_polangle));


    Vector3d V_p;
    if(point[2] > 0){
        V_p = {v_pol*sinz*cos(phi), v_pol*sinz*sin(phi), v_pol*cosz};
    } else {
        V_p = {v_pol*sinz*cos(phi), v_pol*sinz*sin(phi), -v_pol*cosz};
    }
    // FIXME: Move in if(z > 0) to account for the different direction of the rotation?
    Vector3d V_phi = {-sin(phi)*v_phi, cos(phi)*v_phi, 0.0};

    return V_p + V_phi;

};


AccelParabolicVField::AccelParabolicVField(double gamma_0, double gamma_1, Geometry* geometry, double betac_phi, Vector3d origin) :
        VField(geometry), gamma_0_(gamma_0), gamma_1_(gamma_1), betac_phi_(betac_phi), origin_(std::move(origin)) {}

Vector3d AccelParabolicVField::vf(const Vector3d &point) const {
    double x = point[0];
    double y = point[1];
    double z = abs(point[2]);
    double r_cur = hypot(x, y);

    double phi = atan2(y, x);
    // As atan2 returns [-pi, pi], put this to [0, 2pi]
    if(phi < 0){
        phi += 2.0*M_PI;
    }

    double r_border = geometry_->radius_at_given_distance(point);
    double gamma = gamma_0_ + gamma_1_*pow(r_cur/pc, 0.5);
    if(is_profile_set) {
        gamma *= density_profile(r_cur/r_border, r1_, r2_, K1_, K2_, b1_, b2_, b3_);
    }

    double v_phi = c*betac_phi_;
    double v_pol = c*sqrt(gamma*gamma - 1.0)/gamma;

    // Constant in the parabolic equation for a given streamline
    double R_1_cur = r_cur/ sqrt(z);
    // Derivative of the local parabola equation at given point to find the tangent of the poloidal angle
    double tan_polangle = 0.5*R_1_cur/ sqrt(z);

    double sinz = sqrt(tan_polangle*tan_polangle/(1.0 + tan_polangle*tan_polangle));
    double cosz = sqrt(1.0/(1.0 + tan_polangle*tan_polangle));


    Vector3d V_p;
    if(point[2] > 0){
        V_p = {v_pol*sinz*cos(phi), v_pol*sinz*sin(phi), v_pol*cosz};
    } else {
        V_p = {v_pol*sinz*cos(phi), v_pol*sinz*sin(phi), -v_pol*cosz};
    }
    // FIXME: Move in if(z > 0) to account for the different direction of the rotation?
    Vector3d V_phi = {-sin(phi)*v_phi, cos(phi)*v_phi, 0.0};

    return V_p + V_phi;

};