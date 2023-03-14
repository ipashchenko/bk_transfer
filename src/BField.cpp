#include <cmath>
#include <Eigen/Eigen>
#include <BField.h>
#include <utils.h>
#include <boost/math/special_functions/bessel.hpp>

using Eigen::Vector3d;


ScalarBField::ScalarBField(Geometry *geometry_out, Geometry *geometry_in, VField* vfield, bool in_plasma_frame) :
    in_plasma_frame_(in_plasma_frame)
{
    geometry_in_ = geometry_in;
    geometry_out_ = geometry_out;
    vfield_ = vfield;
}


ScalarBField::ScalarBField(ScalarBField *pField) {
	in_plasma_frame_ = pField->in_plasma_frame_;
	geometry_in_ = pField->geometry_in_;
	geometry_out_ = pField->geometry_out_;
	vfield_ = pField->vfield_;
}

// Always specified in plasma frame
double ScalarBField::bf_plasma_frame(const Vector3d &point, Vector3d &v, double t) const {
    double Gamma = getG(v);
    if (in_plasma_frame_) {
        return bf(point, t);
    } else {
        return bf(point, t)/Gamma;
    }
}

double ScalarBField::bf(const Vector3d &point, double t) const {
    double r_border;
    double x = point[0];
    double y = point[1];
    double r_point = sqrt(x*x + y*y);

    double factor = 1.0;

    if(geometry_out_) {
        // Find radius of outer surface at given point
        r_border = geometry_out_->radius_at_given_distance(point);
        if (r_point > r_border) {
            factor = exp(-pow(r_point - r_border, 2.0)/l_eps_B/l_eps_B);
        }
    }
    if(geometry_in_) {
        // Find radius of inner surface at given point
        r_border = geometry_in_->radius_at_given_distance(point);
        if (r_point < r_border) {
            factor = exp(-pow(r_point - r_border, 2.0)/l_eps_B/l_eps_B);
        }
    }
    return _bf(point)*factor;
}

BKScalarBField::BKScalarBField(double b_0, double m_b, Geometry* geometry_out, Geometry* geometry_in, VField* vfield,
                               bool in_plasma_frame) :
        ScalarBField(geometry_out, geometry_in, vfield, in_plasma_frame), b_0_(b_0), m_b_(m_b) {}

double BKScalarBField::_bf(const Vector3d &point, double t) const {
    double r = point.norm();
    return b_0_ * pow(r/pc, -m_b_);
};



FlareBKScalarBField::FlareBKScalarBField(double b_0, double m_b, double amp, double t_start, double width_pc, double theta_los,
                                         double z, Geometry *geometry_out, Geometry *geometry_in, VField *vfield,
                                         bool in_plasma_frame) :
        BKScalarBField(b_0, m_b, geometry_out, geometry_in, vfield, in_plasma_frame),
        amp_(amp),
        t_start_(t_start),
        width_pc_(width_pc),
        theta_los_(theta_los),
        z_(z)
        {}


//double FlareBKScalarBField::_bf(const Vector3d &point, const double t) const {
//    // Direction to the observer
//    Vector3d n_los = {sin(theta_los_), 0, cos(theta_los_)};
//    Vector3d v = vfield_->vf(point);
//    Vector3d v_hat = v.normalized();
//    double cos_theta_local = v_hat.dot(n_los);
//    double beta = v.norm()/c;
//    double beta_app = beta/(1.0 - beta*cos_theta_local)/(1.0 + z_);
//
//    double r = point.norm();
//    return BKScalarBField::_bf(point, t) * exp(-pow(r - beta_app*c*(t - t_start_), 2.0)/(width_pc_*width_pc_*pc*pc));
//}


double FlareBKScalarBField::_bf(const Vector3d &point, const double t) const {
    // Direction to the observer
//    Vector3d n_los = {sin(theta_los_), 0, cos(theta_los_)};
    Vector3d v = vfield_->vf(point);
    double r = point.norm();
    return amp_*BKScalarBField::_bf(point, t) * exp(-pow(r - v.norm()*(t - t_start_), 2.0)/(width_pc_*width_pc_*pc*pc));
}



FlareBKBField::FlareBKBField(ScalarBField* bkg_bfield, double amp, double t_start, double width_pc, VField* flare_pattern_vfield,
							 Geometry* geometry_out, Geometry* geometry_in) :
	 ScalarBField(bkg_bfield),
	 amp_(amp),
	 t_start_(t_start),
	 width_pc_(width_pc)
	{
		bkg_bfield_ = bkg_bfield;
		flare_pattern_vfield_ = flare_pattern_vfield;
		if(geometry_out)
		{
			geometry_out_ = geometry_out;
		}
		if(geometry_in)
		{
			geometry_in_ = geometry_in;
		}
	}
	

double FlareBKBField::_bf(const Vector3d &point, double t) const {
	// TODO: Here we should integrate flare pattern speed in time to find the position of the flare (e.g. blob center).
	// We can model flare as a spherical blob.
	Vector3d v = flare_pattern_vfield_->vf(point);
	double r = point.norm();
	//	return amp_ * bkg_nfield_->_nf(point, t) * exp(-pow(r - v.norm()*(t - t_start_), 2.0)/(width_pc_*width_pc_*pc*pc));
	return amp_ * bkg_bfield_->_bf(point, t) * generalized1_gaussian1d(r, v.norm()*(t - t_start_), width_pc_*pc, 10.0);
}


VectorBField::VectorBField(bool in_plasma_frame, double tangled_fraction, Geometry* geometry_out, Geometry* geometry_in) :
    in_plasma_frame_(in_plasma_frame),
    tangled_fraction_(tangled_fraction) {
    geometry_in_ = geometry_in;
    geometry_out_ = geometry_out;
}

double VectorBField::get_tangled_fraction(const Vector3d &point, double t) const {
    return tangled_fraction_;
}

Vector3d VectorBField::bf(const Vector3d &point, double t) const {
    double x = point[0];
    double y = point[1];
    double r_point = sqrt(x*x + y*y);
//    std::cout << "r = " << r_point/pc << "\n";

    if(geometry_out_) {
        // Find radius of outer surface at given point
        double r_border_out = geometry_out_->radius_at_given_distance(point);
//        std::cout << "r_border = " << r_border_out/pc << "\n";
        if (r_point >= r_border_out) {
            return {0.0, 0.0, 0.0};
        }
    }
    if(geometry_in_) {
        // Find radius of inner surface at given point
        std::cout << "NEVER =========================================== " << "\n";
        double r_border_in = geometry_in_->radius_at_given_distance(point);
        if (r_point < r_border_in) {
            return {0.0, 0.0, 0.0};
        }
    }
    return _bf(point, t);
}

Vector3d VectorBField::bf_plasma_frame(const Vector3d &point, Vector3d &v, double t) const {
    Vector3d b = bf(point, t);
    if (in_plasma_frame_) {
        return b;
    } else {
        return get_B_prime(b, v);
    }
}

double VectorBField::bf_tangled_plasma_frame(const Vector3d &point, Vector3d &v, double t) const {
    Vector3d b = bf(point, t);
    if (tangled_fraction_ > 0.0) {
        if (in_plasma_frame_) {
            return tangled_fraction_*b.norm();
        } else {
            return tangled_fraction_*get_B_prime(b, v).norm();
        }
    } else {
        return 0.0;
    }
}

Vector3d VectorBField::bhat_lab_frame(const Vector3d &point, Vector3d &v, double t) const {
    Vector3d b = bf(point, t);
    if (in_plasma_frame_) {
        auto b_hat_prime = b.normalized();
        Vector3d beta = v/c;
        return get_B_hat(b_hat_prime, beta);
    } else {
        return b.normalized();
    }
}


ConstCylinderBFieldZ::ConstCylinderBFieldZ(double b_0, double n_b, bool in_plasma_frame, double tangled_fraction, Geometry* geometry_out, Geometry* geometry_in) :
    VectorBField(in_plasma_frame, tangled_fraction, geometry_out, geometry_in), b_0_(b_0), n_b_(n_b) {};

Vector3d ConstCylinderBFieldZ::_bf(const Vector3d &point, double t) const {
    double r = abs(point[2]);
    double z = point[2];
    if(z > 0) {
        return Vector3d(0.0, 0.0, b_0_*pow(r/pc, -n_b_));
    } else {
        return Vector3d(0.0, 0.0, -b_0_*pow(r/pc, -n_b_));
    }
}


//RadialConicalBField::RadialConicalBField(double b_0, double n_b, bool in_plasma_frame, double tangled_fraction) :
//    VectorBField(in_plasma_frame, tangled_fraction), b_0_(b_0), n_b_(n_b) {};
//
//Vector3d RadialConicalBField::bf(const Vector3d &point) const {
//    double r = point.norm();
//    double z = point[2];
//    return Vector3d(b_0_*pow(r/pc, -n_b_)*point[0]/r,
//                    b_0_*pow(r/pc, -n_b_)*point[1]/r,
//                    b_0_*pow(r/pc, -n_b_)*point[2]/r);
//}
//


ToroidalBField::ToroidalBField(double b_0, double n_b, bool in_plasma_frame, double tangled_fraction, Geometry* geometry_out, Geometry* geometry_in) :
    VectorBField(in_plasma_frame, tangled_fraction, geometry_out, geometry_in), b_0_(b_0), n_b_(n_b){};

Vector3d ToroidalBField::_bf(const Vector3d &point, double t) const {
    double x = point[0];
    double y = point[1];
    double z = point[2];
    double fi = atan2(y, x);
    double b = b_0_*pow(z/pc, -n_b_);
    return {-sin(fi)*b, cos(fi)*b, 0};
}


// TODO: Check helicity (rotation direction) for z < 0
HelicalCylinderBField::HelicalCylinderBField(double b_0, double pitch_angle, bool in_plasma_frame, double tangled_fraction, Geometry* geometry_out, Geometry* geometry_in) :
    VectorBField(in_plasma_frame, tangled_fraction, geometry_out, geometry_in), b_0_(b_0), pitch_angle_(pitch_angle) {};

Vector3d HelicalCylinderBField::_bf(const Vector3d &point, double t) const {
    double z = point[2];
    double phi = atan2(point[1], point[0]);
    double b_z = b_0_*cos(pitch_angle_);
    // FIXME: Here was + sign
    double b_phi = b_0_*sin(pitch_angle_);
    if(z > 0) {
        return b_phi*Vector3d(-sin(phi), cos(phi), 0) + b_z*Vector3d(0, 0, -1);
    } else {
        return b_phi*Vector3d(-sin(phi), cos(phi), 0) + b_z*Vector3d(0, 0, -1);
    }
}


// TODO: Check helicity (rotation direction) for z < 0
HelicalConicalBField::HelicalConicalBField(double b_0, double n_b, double pitch_angle, bool in_plasma_frame, double tangled_fraction, Geometry* geometry_out, Geometry* geometry_in) :
    VectorBField(in_plasma_frame, tangled_fraction, geometry_out, geometry_in), b_0_(b_0), n_b_(n_b), pitch_angle_(pitch_angle) {};

Vector3d HelicalConicalBField::_bf(const Vector3d &point, double t) const {
    double z = point[2];
    double b = b_0_*pow(abs(z)/pc, -n_b_);
    double phi = atan2(point[1], point[0]);
    double b_z = b*cos(pitch_angle_);
    double b_phi = b*sin(pitch_angle_);
    if(z > 0) {
        return b_phi*Vector3d(-sin(phi), cos(phi), 0) + b_z*Vector3d(0, 0, 1);
    } else {
        return b_phi*Vector3d(-sin(phi), cos(phi), 0) + b_z*Vector3d(0, 0, -1);
    }
}


//SpiralConicalBField::SpiralConicalBField(double b_0, double pitch_angle, bool in_plasma_frame, double tangled_fraction) :
//    VectorBField(in_plasma_frame, tangled_fraction), b_0_(b_0), pitch_angle_(pitch_angle) {};
//
//Vector3d SpiralConicalBField::bf(const Vector3d &point) const {
//    double z = point[2];
//    double x = point[0];
//    double y = point[1];
//    double b_z = b_0_/(z*z/(pc*pc));
//    return Vector3d(b_z*(x/z + y*tan(pitch_angle_)/pc),
//                    b_z*(y/z - x*tan(pitch_angle_)/pc),
//                    b_z);
//}
//
//ForceFreeCylindricalBField::ForceFreeCylindricalBField(double b_0, double mu, bool in_plasma_frame, double tangled_fraction) :
//    VectorBField(in_plasma_frame, tangled_fraction), b_0_(b_0), mu_(mu) {};
//
//Vector3d ForceFreeCylindricalBField::bf(const Vector3d &point) const {
//    double x = point[0];
//    double y = point[1];
//    double atan_term = atan(y/x);
//    double bessel_0 = boost::math::cyl_bessel_i(0, mu_);
//    double bessel_1 = boost::math::cyl_bessel_i(1, mu_);
//    return Vector3d(-b_0_*bessel_1*sin(atan_term),
//                    b_0_*bessel_1*sin(atan_term),
//                    b_0_*bessel_0);
//}


ReversedPinchCylindricalBField::ReversedPinchCylindricalBField(double b_0, double tangled_fraction,
                                                               Geometry* geometry_out, Geometry* geometry_in) :
    VectorBField(true, tangled_fraction, geometry_out, geometry_in), b_0_(b_0) {};

Vector3d ReversedPinchCylindricalBField::_bf(const Vector3d &point, double t) const {
    double x = point[0];
    double y = point[1];
    double phi = atan2(y, x);
    double r_border = geometry_out_->radius_at_given_distance(point);
    double ro_normed = sqrt(point[0]*point[0]+ point[1]*point[1])/r_border;

    double bessel_0 = boost::math::cyl_bessel_i(0, 2.405*ro_normed);
    double bessel_1 = boost::math::cyl_bessel_i(1, 2.405*ro_normed);
    auto B_cyl =  Vector3d(0, b_0_*bessel_1,b_0_*bessel_0);

    Eigen::Matrix3d cyl_to_cart;
    cyl_to_cart << cos(phi), -sin(phi), 0,
                   sin(phi), cos(phi), 0,
                   0, 0, 1;
    return cyl_to_cart*B_cyl;
}


ReversedPinchConicalBField::ReversedPinchConicalBField(double b_0, double n_b, Geometry* geometry, double tangled_fraction) :
    VectorBField(true, tangled_fraction), b_0_{b_0}, n_b_(n_b) {
    geometry_ = geometry;
}

Vector3d ReversedPinchConicalBField::_bf(const Vector3d &point, double t) const {
    double x = point[0];
    double y = point[1];
    double z = point[2];
    double b = b_0_*pow(z/pc, -n_b_);
    double phi = atan2(y, x);
    // Find radius at given point
    double r_border = geometry_->radius_at_given_distance(point);
    double ro_normed = sqrt(x*x + y*y)/r_border;

    double bessel_0 = boost::math::cyl_bessel_i(0, 2.405*ro_normed);
    double bessel_1 = boost::math::cyl_bessel_i(1, 2.405*ro_normed);
    auto B_cyl =  Vector3d(0, b*bessel_1,b*bessel_0);

    Eigen::Matrix3d cyl_to_cart;
    cyl_to_cart << cos(phi), -sin(phi), 0,
        sin(phi), cos(phi), 0,
        0, 0, 1;

    return cyl_to_cart*B_cyl;
}