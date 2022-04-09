#include <cmath>
#include <string>
#include <iostream>
#include <fstream>
#include "utils.h"
#include "MyExceptions.h"


double nu_p(double n) {return sqrt(n*q_e*q_e / (pi*m_e));}

double nu_b(Vector3d &b, Vector3d &n_los) {
    return q_e*(n_los.cross(b)).norm()/(2.*pi*m_e*c);
}

double nu_b(double b) {
	return q_e*b/(2.*pi*m_e*c);
}

double nu_min(double gamma_min, Vector3d &b, Vector3d &n_los) {
    return gamma_min*gamma_min*nu_b(b, n_los);
}

double nu_min(double gamma_min, double b) {
    return gamma_min*gamma_min*nu_b(b);
}

double sin_theta(Vector3d &b, Vector3d &n_los) {
		return n_los.cross(b).norm()/b.norm();
};

double nu_b_value(Vector3d &b) {
	return q_e*b.norm()/(2.*pi*m_e*c);
}

double k_0(Vector3d &b, Vector3d &n_los, double nu, double n_nt, double s, double gamma_min, double gamma_max) {
//    double n = n_nt*(s-1)*pow(gamma_min, s-1);
    double n = n_nt*(s-1)/(pow(gamma_min, 1.-s) - pow(gamma_max, 1.-s));
    return pi*nu_p(n)*nu_p(n)*nu_b(b, n_los)/(c*nu*nu);
}


double k_0(double b, Vector3d &n_los, double nu, double n_nt, double s, double gamma_min, double gamma_max) {
//    double n = n_nt*(s-1)*pow(gamma_min, s-1);
    double n = n_nt*(s-1)/(pow(gamma_min, 1.-s) - pow(gamma_max, 1.-s));
    return pi*nu_p(n)*nu_p(n)*nu_b(b)/(c*nu*nu);
}

double k_0_value(Vector3d &b, double nu, double n_nt, double s, double gamma_min, double gamma_max) {
//    double n = n_nt*(s-1)*pow(gamma_min, s-1);
    double n = n_nt*(s-1)/(pow(gamma_min, 1.-s) - pow(gamma_max, 1.-s));
    return pi*nu_p(n)*nu_p(n)*nu_b_value(b)/(c*nu*nu);
}

double eta_0(Vector3d &b, Vector3d &n_los, double n_nt, double s, double gamma_min, double gamma_max) {
//    double n = n_nt*(s-1)*pow(gamma_min, s-1);
    double n = n_nt*(s-1)/(pow(gamma_min, 1.-s) - pow(gamma_max, 1.-s));
    return pi*nu_p(n)*nu_p(n)*nu_b(b, n_los)*m_e/c;
}

double eta_0(double b, Vector3d &n_los, double n_nt, double s, double gamma_min, double gamma_max) {
//    double n = n_nt*(s-1)*pow(gamma_min, s-1);
    double n = n_nt*(s-1)/(pow(gamma_min, 1.-s) - pow(gamma_max, 1.-s));
    return pi*nu_p(n)*nu_p(n)*nu_b(b)*m_e/c;
}

double eta_0_value(Vector3d &b, double n_nt, double s, double gamma_min, double gamma_max) {
//    double n = n_nt*(s-1)*pow(gamma_min, s-1);
    double n = n_nt*(s-1)/(pow(gamma_min, 1.-s) - pow(gamma_max, 1.-s));
    return pi*nu_p(n)*nu_p(n)*nu_b_value(b)*m_e/c;
}

double getG(Vector3d &v) {
    Vector3d beta = v/c;
    return sqrt(1./(1.- beta.squaredNorm()));
}

double getD(Vector3d &n_los, Vector3d &v) {
    Vector3d beta = v/c;
//    double angle = acos(beta.dot(n_los))*180.0/M_PI;
//    if(angle < 3.0){
//        std::cout << "LOS angle = " << angle << "\n";
//    }
    return 1./(getG(v)*(1.-beta.dot(n_los)));
}

// (3) from doi:10.1088/0004-637X/695/1/503 (Gracia), (A16) from doi:10.1088/0004-637X/737/1/42 (Porth)
Vector3d get_n_los_prime(Vector3d &n_los, Vector3d &v) {
    double df = getD(n_los, v);
    double gamma = getG(v);
    Vector3d beta = v/c;
    return df*n_los - (df+1.)*(gamma/(gamma+1.))*beta;
}

// (4) in Lyutikov 2003
//Vector3d get_n_los_prime(Vector3d &n_los, Vector3d &v) {
//    double gamma = getG(v);
//    Vector3d beta = v/c;
//    return (n_los + gamma*beta*(gamma*n_los.dot(beta)/(gamma+1)-1))/(gamma*(1-n_los.dot(beta)));
//}

// FIXME: Check this two functions: choose some b_lab, use get_B_prime to obtain b_plasma, then use get_B_hat to obtain
// the direction of b_lab. Compare with the original value!
// (4) from doi:10.1088/0004-637X/695/1/503 (Gracia); (4.33) from Daniel, Herbert (1997), "Physik: Elektrodynamik,
// relativistische Physik"
// TODO: Formally, for b = b1 + b2, get_B_prime(b, v) = get_B_prime(b1, v) + get_B_prime(b2, v). Is it OK physically?
Vector3d get_B_prime(Vector3d &b, Vector3d &v) {
    double gamma = getG(v);
    Vector3d result = b/gamma + gamma/((1.+gamma)*(c*c))*v*v.dot(b);
    return result;
}


Vector3d get_B_hat(Vector3d &b_hat_prime, Vector3d &v_beta) {
    double Gamma = 1.0/sqrt(1.0-v_beta.squaredNorm());
    double bv = b_hat_prime.dot(v_beta);
    return (1./sqrt(1.0-bv*bv))*(b_hat_prime - (Gamma/(Gamma+1.0))*bv*v_beta);
}


// from doi:10.1088/0004-637X/737/1/42 (Porth)
//Vector3d get_B_prime(Vector3d &b, Vector3d &v) {
//    double gamma = getG(v);
//    Vector3d result = b/gamma + gamma*v*v.dot(b)/(c*c);
//    return result;
//}

Vector3d q(Vector3d &b_hat, Vector3d &v_beta, Vector3d &n_hat) {
    if (v_beta.norm() == 0){
        return b_hat;
    } else {
        return b_hat + n_hat.cross(v_beta.cross(b_hat));
    }
}

Vector3d e_hat(Vector3d &b_hat, Vector3d &v_beta, Vector3d &n_hat) {
    auto q_ = q(b_hat, v_beta, n_hat);
    auto nq = n_hat.dot(q_);
    // FIXME: Sometimes n_hat * q_ is a bit (1E-17) larger than q_^2!
//    return n_hat.cross(q_)/sqrt(q_.squaredNorm() - nq*nq);
    return n_hat.cross(q_)/sqrt(abs(q_.squaredNorm() - nq*nq));
}

// This uses adaptive integration as the Astropy version with the same tolerances.
double comoving_transfer_distance2(double z, double H0, double omega_M, double omega_V, double gamma_nu) {
    Ctd ctd(z, H0, omega_M, omega_V, gamma_nu);
    double ctd_state = 0.0;

    using namespace boost::numeric::odeint;
    typedef runge_kutta_dopri5<double> stepper_type;
    auto stepper = stepper_type();
    integrate_adaptive(make_controlled(1.49e-8, 1.49e-8, stepper_type()), ctd, ctd_state, 0.0, z, 1E-09);

    double result = (299792.458/H0) * pow(10., 6.) * ctd_state;
    return result;
}

double pc_to_mas(double z) {
	double d_a = comoving_transfer_distance2(z)/(1.+z);
	double angle_rads = 1./d_a;
	return rad_to_mas*angle_rads;
}

double mas_to_pc(double z) {
	double d_a = comoving_transfer_distance2(z)/(1.+z);
	return mas_to_rad*d_a;
}

std::ostream &
write_2dvector(std::ostream &os, std::vector<std::vector<double>> &v,
							 double scale) {
	for (auto & i : v) {
		for (double j : i) {
			double value = j/scale;
			os << value <<" ";
		}
		os<<"\n";
	}
	return os;
}

Ctd::Ctd(double z, double H0, double omega_M, double omega_V, double gamma_nu) : z(z), H0(H0), omega_M(omega_M), omega_V(omega_V), gamma_nu(gamma_nu) {}

void Ctd::operator()(const double &x, double &dxdt, const double t) {
	dxdt = pow((omega_M + (1.+t)*gamma_nu)*(1.+t)*(1.+t)*(1.+t)+omega_V, -0.5);
	//dxdt = pow(omega_M*(1.+t*t*t)+omega_V, -0.5);
};


void read_from_txt(const std::string& fntxt, std::vector< std::vector<double> >& properties) {
    std::ifstream infile(fntxt);
    if(!infile.good()){
        throw AbsentDataFile(fntxt);
    }
    std::vector<double> row(3);
    while (infile >> row[0] >> row[1] >> row[2])
    {
        properties.push_back(row);
    }
}


double generalized1_gaussian1d(double x, double loc, double scale, double shape) {
//    return shape/(2*scale*tgamma(1/shape))*exp(-pow(abs(x-loc)/scale, shape));
    return exp(-pow(abs(x-loc)/scale, shape));
}

double density_profile(double r, double r1, double r2, double K1, double K2, double b1, double b2, double b3) {
    auto l1 = tanh(K1*(r-r1));
    auto l2 = tanh(K2*(r-r2));
    // b1 at r < r1
    // At r < r2:
    auto y1 = b1 + 0.5*(1+l1)*(b2-b1);
    // For all r
    auto y2 = y1 + 0.5*(1+l2)*(b3-y1);
    return y2;
}


double distance_plane_point(const Vector3d& P, const Vector3d& P_plane, const Vector3d& N_plane) {
    return (P - P_plane).dot(N_plane.normalized());
}
