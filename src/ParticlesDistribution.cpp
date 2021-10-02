#include "ParticlesDistribution.h"
#include "utils.h"
#include "MyExceptions.h"


double ParticlesDistribution::k_F_c(Vector3d &b, Vector3d &n_los, double nu, double n) const {
    double cos_theta = b.dot(n_los)/b.norm();
    return 2.*pi*nu_p(n)*nu_p(n)*nu_b_value(b)*cos_theta/(c*nu*nu);
}

double ParticlesDistribution::k_C_c(Vector3d &b, Vector3d &n_los, double nu, double n) const {
    auto nu_b_calc = nu_b(b, n_los);
    return -pi*nu_p(n)*nu_p(n)*nu_b_calc*nu_b_calc/(c*nu*nu*nu);
}

PowerLaw::PowerLaw(double s, double gamma_min, std::string plasma, bool changing_s, double ds)
    : s_(s), gamma_min_(gamma_min), plasma_(plasma), changing_s_(changing_s), ds_(ds) {
    if (plasma_=="normal") {
        std::cout << "Normal plasma content with gamma_min = " << gamma_min_ << std::endl;
    } else if (plasma_=="pairs") {
        std::cout << "Pairs plasma content with gamma_min = " << gamma_min_ << std::endl;
    } else {
        throw BadPlasmaContentParameters();
    }
    factor_ki_ = (pow(3., (s_ + 1.)/2.)/4.)*tgamma(s_/4. + 11./6.)*tgamma(s_/4. + 1./6.);
    factor_ki_rnd_ = sqrt(pi/4.)*tgamma((6. + s_)/4.)/tgamma((8. + s_)/4.);
    factor_kv_ = pow(3., s_/2.)*(s_ + 3.)*(s_ + 2.)/(4.*(s_ + 1.))*tgamma(s_/4. + 11./12.)*tgamma(s_/4. + 7./12.);
    factor_kf_ = (s_ + 2.)*log(gamma_min_)/((s_ + 1.)*pow(gamma_min_, s_ + 1.));
    factor_etai_ = pow(3., s_/2.)/(2.*(s_ + 1))*tgamma(s_/4. + 19./12.)*tgamma(s_/4. - 1./12.);
    factor_etai_rnd_ = sqrt(pi/4.)*tgamma((5. + s_)/4.)/tgamma((7. + s_)/4.);
    factor_etav_ = pow(3., (s_ - 1.)/2.)*(s_ + 2.)/(2.*s_)*tgamma(s_/4. + 2./3.)*tgamma(s_/4. + 1./3.);
    if (plasma_=="pairs") {
        factor_kf_ = 0.0;
        factor_kv_ = 0.0;
        factor_etav_ = 0.0;
    }
}

double PowerLaw::get_s(const Vector3d &point) const {
    double r = point.norm();
    return s_ + ds_*r/pc;
}

double PowerLaw::get_s(Vector3d &b, Vector3d &n_los) const {
    double s_along = 2;
    double s_across = 4;
    double cos_theta = abs(b.dot(n_los)/b.norm());
    return s_along*cos_theta + s_across*(1 - cos_theta);
}

double PowerLaw::k_i(Vector3d &b, Vector3d &n_los, double nu, double n_nt) const {
    // If only scalar field is supplied
    if(b.norm() < eps_B) {
        return 0.0;
    }
    double factor_ki;
    double s;
    if (changing_s_) {
        s = get_s(b, n_los);
        factor_ki = (pow(3., (s + 1.)/2.)/4.)*tgamma(s/4. + 11./6.)*tgamma(s/4. + 1./6.);
    } else {
        s = s_;
        factor_ki = factor_ki_;
    }

    double nu_min_ = nu_min(gamma_min_, b, n_los);
    if(nu > nu_min_) {
        return k_0(b, n_los, nu, n_nt, s, gamma_min_) * pow(nu_b(b, n_los)/nu, s/2.) * factor_ki;
//        return k_0(b, n_los, nu, n*(s-1)*pow(gamma_min_, s-1)) * pow(nu_b(b, n_los)/nu, s/2.) * factor_ki;
    } else {
        return k_0(b, n_los, nu_min_, n_nt, s, gamma_min_) * pow(nu_b(b, n_los)/nu_min_, s/2.) * factor_ki * pow(nu/nu_min_, -5./3.);
//        return k_0(b, n_los, nu_min_, n*(s-1)*pow(gamma_min_, s-1)) * pow(nu_b(b, n_los)/nu_min_, s/2.) * factor_ki * pow(nu/nu_min_, -5./3.);
    }


}

double PowerLaw::k_i(double b, Vector3d &n_los, double nu, double n_nt) const {
    if (b == 0.0) {
        return 0.0;
    }
    //double factor = (pow(3., (s_+1.)/2.)/4.)*tgamma(s_/4.+11./6.)*tgamma(s_/4.+1./6.);
    //double rnd_factor = sqrt(pi/4.)*tgamma((6.+s_)/4.)/tgamma((8.+s_)/4.);
    double factor = factor_ki_*factor_ki_rnd_;
    double nu_min_ = nu_min(gamma_min_, b);
    if (nu > nu_min_) {
//        return k_0(b, n_los, nu, n)*pow(nu_b(b)/nu, s_/2.)*factor;
        return k_0(b, n_los, nu, n_nt, s_, gamma_min_) * pow(nu_b(b)/nu, s_/2.) * factor;

    } else {
        //std::cout << "nu_min[GHz] = " << nu_min_/1E+09 << ", while nu[GHz] = " << nu/1E+09 << std::endl;
        return k_0(b, n_los, nu_min_, n_nt, s_, gamma_min_)*pow(nu_b(b)/nu_min_, s_/2.)*factor*pow(nu/nu_min_, -5./3.);
    }
}

double PowerLaw::k_q(Vector3d &b, Vector3d &n_los, double nu, double n_nt) const {
    // If only scalar field is supplied
    if(b.norm() < eps_B) {
        return 0.0;
    }
    double s;
    if (changing_s_) {
        s = get_s(b, n_los);
    } else {
        s = s_;
    }
    return (s + 2.)/(s + 10./3)*k_i(b, n_los, nu, n_nt);
}

double PowerLaw::k_u(Vector3d &b, Vector3d &n_los, double nu, double n_nt) const {
    return 0;
}

double PowerLaw::k_v(Vector3d &b, Vector3d &n_los, double nu, double n_nt) const {
    // If only scalar field is supplied
    if(b.norm() < eps_B) {
        return 0.0;
    }
    if (plasma_ == "pairs") {
        return 0.0;
    }

    double factor_kv;
    double s;
    if (changing_s_) {
        s = get_s(b, n_los);
        factor_kv = pow(3., s/2.)*(s + 3.)*(s + 2.)/(4.*(s + 1.))*tgamma(s/4. + 11./12.)*tgamma(s/4. + 7./12.);
    } else {
        s = s_;
        factor_kv = factor_kv_;
    }

    double cos_theta = b.dot(n_los)/b.norm();
    //double factor = pow(3., s_/2.)*(s_+3.)*(s_+2.)/(4.*(s_+1.))*tgamma(s_/4.+11./12.)*tgamma(s_/4.+7./12.);
    double nu_min_ = nu_min(gamma_min_, b, n_los);
    if (nu > nu_min_) {
        return -k_0_value(b, nu, n_nt, s, gamma_min_)*cos_theta*pow(nu_b(b, n_los)/nu, (s + 1.)/2.)*factor_kv;
    } else {
        return -k_0_value(b, nu_min_, n_nt, s, gamma_min_)*cos_theta*pow(nu_b(b, n_los)/nu_min_, (s + 1.)/2.)*factor_kv*pow(nu/nu_min_, -5./3.);
    }
}

double PowerLaw::k_F(Vector3d &b, Vector3d &n_los, double nu, double n_nt) const {
    // If only scalar field is supplied
    if(b.norm() < eps_B) {
        return 0.0;
    }
    if (plasma_ == "pairs") {
        return 0.0;
    }
    double factor_kf;
    double s;
    if (changing_s_) {
        s = get_s(b, n_los);
        factor_kf = (s + 2.)*log(gamma_min_)/((s + 1.)*pow(gamma_min_, s + 1.));
    } else {
        factor_kf = factor_kf_;
    }
    //return (s_+2.)*log(gamma_min_)/((s_+1.)*pow(gamma_min_, s_+1.))*k_F_c(b, n_los, nu, n);
    return factor_kf*k_F_c(b, n_los, nu, n_nt);
}

double PowerLaw::k_C(Vector3d &b, Vector3d &n_los, double nu, double n_nt) const {
    // If only scalar field is supplied
    if(b.norm() < eps_B) {
        return 0.0;
    }
    double s;
    if (changing_s_) {
        s = get_s(b, n_los);
    } else {
        s = s_;
    }
    if(s != 2.0) {
        return (2./(s - 2.))*(pow(gamma_min_, 2. - s) - pow(nu_b(b, n_los)/nu, (s - 2.)/2.))*k_C_c(b, n_los, nu, n_nt);
    }
    // Beckert & Falke, A&A 338, 2002
    else {
        double gamma_rad = sqrt(nu/nu_b(b, n_los));
        return 2.*pow(gamma_min_, 2. - s)*log(gamma_rad/gamma_min_)*k_C_c(b, n_los, nu, n_nt);
    }
}

// Debug s = 2 case
//double PowerLaw::k_C(Vector3d &b, Vector3d &n_los, double nu, double n_nt) const {
//    double s;
//    if (changing_s_) {
//        s = get_s(b, n_los);
//    } else {
//        s = s_;
//    }
//    double result = (2./(s - 2.))*(pow(gamma_min_, 2. - s) - pow(nu_b(b, n_los)/nu, (s - 2.)/2.))*k_C_c(b, n_los, nu, n_nt);
//    if(isnan(result)){
//        std::cout << "NaN in K_C_r! ========= " << std::endl;
//        double k_c_c = k_C_c(b, n_los, nu, n_nt);
//        std::cout << "k_C_c = " << k_c_c << "\n";
//        double c1 = pow(gamma_min_, 2. - s);
//        double c2 = pow(nu_b(b, n_los)/nu, (s - 2.)/2.);
//        std::cout << "c1 = " << c1 << ", c2 = " << c2 << '\n';
//
//    }
//    return result;
//}


double PowerLaw::h_Q(Vector3d &b, Vector3d &n_los, double nu, double n_nt) const {
    return 0;
}

double PowerLaw::eta_i(Vector3d &b, Vector3d &n_los, double nu, double n_nt) const {
    //double factor = pow(3., s_/2.)/(2.*(s_+1))*tgamma(s_/4.+19./12.)*tgamma(s_/4.-1./12.);
    // If only scalar field is supplied
    if(b.norm() < eps_B) {
        return 0.0;
    }
    double factor_etai;
    double s;
    if (changing_s_) {
        s = get_s(b, n_los);
        factor_etai = pow(3., s/2.)/(2.*(s + 1))*tgamma(s/4. + 19./12.)*tgamma(s/4. - 1./12.);
    } else {
        s = s_;
        factor_etai = factor_etai_;
    }

    double nu_min_ = nu_min(gamma_min_, b, n_los);
    if (nu > nu_min_) {
        return eta_0(b, n_los, n_nt, s, gamma_min_)*pow(nu_b(b, n_los)/nu, (s - 1.)/2.)*factor_etai;
    } else {
        return eta_0(b, n_los, n_nt, s, gamma_min_)*pow(nu_b(b, n_los)/nu_min_, (s - 1.)/2.)*factor_etai*pow(nu/nu_min_, 1./3.);
    }
}

double PowerLaw::eta_i(double b, Vector3d &n_los, double nu, double n_nt) const {
    if (b == 0.0) {
        return 0.0;
    }
    //double factor = pow(3., s_/2.)/(2.*(s_+1))*tgamma(s_/4.+19./12.)*tgamma(s_/4.-1./12.);
    //double rnd_factor = sqrt(pi/4.)*tgamma((5.+s_)/4.)/tgamma((7.+s_)/4.);
    double factor = factor_etai_*factor_etai_rnd_;
    double nu_min_ = nu_min(gamma_min_, b);

    if (nu > nu_min_) {
        return eta_0(b, n_los, n_nt, s_, gamma_min_)*pow(nu_b(b)/nu, (s_ - 1.)/2.)*factor;
    } else {
        return eta_0(b, n_los, n_nt, s_, gamma_min_)*pow(nu_b(b)/nu_min_, (s_ - 1.)/2.)*factor*pow(nu/nu_min_, 1./3.);
    }
}

double PowerLaw::eta_q(Vector3d &b, Vector3d &n_los, double nu, double n_nt) const {
    // If only scalar field is supplied
    if(b.norm() < eps_B) {
        return 0.0;
    }
    double s;
    if (changing_s_) {
        s = get_s(b, n_los);
    } else {
        s = s_;
    }
    return (s + 1.0)/(s + 7./3.)*eta_i(b, n_los, nu, n_nt);
}

double PowerLaw::eta_u(Vector3d &b, Vector3d &n_los, double nu, double n_nt) const {
    return 0;
}

double PowerLaw::eta_v(Vector3d &b, Vector3d &n_los, double nu, double n_nt) const {
    // If only scalar field is supplied
    if(b.norm() < eps_B) {
        return 0.0;
    }
    if (plasma_ == "pairs") {
        return 0.0;
    }
    double factor_etav;
    double s;
    if (changing_s_) {
        s = get_s(b, n_los);
        factor_etav = pow(3., (s - 1.)/2.)*(s + 2.)/(2.*s)*tgamma(s/4. + 2./3.)*tgamma(s/4. + 1./3.);
    } else {
        s = s_;
        factor_etav = factor_etav_;
    }

    double cos_theta = b.dot(n_los)/b.norm();
    //double factor = pow(3., (s_-1.)/2.)*(s_+2.)/(2.*s_)*tgamma(s_/4.+2./3.)*tgamma(s_/4.+1./3.);

    double nu_min_ = nu_min(gamma_min_, b, n_los);
    if (nu > nu_min_) {
        return -eta_0_value(b, n_nt, s, gamma_min_)*cos_theta*pow(nu_b(b, n_los)/nu, s/2.)*factor_etav;
    } else {
        return -eta_0_value(b, n_nt, s, gamma_min_)*cos_theta*pow(nu_b(b, n_los)/nu_min_, s/2.)*factor_etav*pow(nu/nu_min_, 1./3.);
    }
}


double Thermal::k_i(Vector3d &b, Vector3d &n_los, double nu, double n) const {
    return 0.0;
}

double Thermal::k_i(double b, Vector3d &n_los, double nu, double n) const {
    return 0.0;
}

double Thermal::k_q(Vector3d &b, Vector3d &n_los, double nu, double n) const {
    return 0.0;
}

double Thermal::k_u(Vector3d &b, Vector3d &n_los, double nu, double n) const {
    return 0.0;
}

double Thermal::k_v(Vector3d &b, Vector3d &n_los, double nu, double n) const {
    return 0.0;
}

double Thermal::k_F(Vector3d &b, Vector3d &n_los, double nu, double n) const {
    return k_F_c(b, n_los, nu, n);
}

double Thermal::k_C(Vector3d &b, Vector3d &n_los, double nu, double n) const {
    return k_C_c(b, n_los, nu, n);
}

double Thermal::h_Q(Vector3d &b, Vector3d &n_los, double nu, double n) const {
    return 0.0;
}

double Thermal::eta_i(Vector3d &b, Vector3d &n_los, double nu, double n) const {
    return 0.0;
}

double Thermal::eta_i(double b, Vector3d &n_los, double nu, double n) const {
    return 0.0;
}

double Thermal::eta_q(Vector3d &b, Vector3d &n_los, double nu, double n) const {
    return 0.0;
}

double Thermal::eta_u(Vector3d &b, Vector3d &n_los, double nu, double n) const {
    return 0.0;
}

double Thermal::eta_v(Vector3d &b, Vector3d &n_los, double nu, double n) const {
    return 0.0;
}
