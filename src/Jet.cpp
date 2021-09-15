#include <tuple>
#include "Jet.h"
#include "MyExceptions.h"

using Eigen::Vector3d;


Jet::Jet(BaseGeometry *newgeo, VField *newvfield, std::vector<VectorBField*> newbFields, std::vector<NField*> newnFields) {
    geometry_ = newgeo;
    vfield_ = newvfield;
    bfields_ = newbFields;
    nfields_ = newnFields;
}


std::tuple<double, double, double, double, double, double, double, double, double, double, double> Jet::get_transport_coefficients(Vector3d &point, Vector3d &n_los, double nu) {
    // Example for k_I (Lyutikov et al. 2005):
    // First, comoving frame ``k_i_prime`` (in the rest frame of the emission element) is connected to this ``k_i`` as
    // ``k_i = k_i_prime / D``. Second, in ``k_i_prime`` we need all quantities in comoving frame (primed) in terms of
    // lab frame:
    // b_prime = f(b, v)
    // n_los_prime = f(n_los, v)
    // nu_prime = f(nu, n_los, v) = nu/getD
    // n_prime = f(n, v) = n/Gamma

    Vector3d v = getV(point);
    auto gamma = getG(v);

    Vector3d b_prime{0.0, 0.0, 0.0};
    Vector3d local_b_prime{0.0, 0.0, 0.0};
    double b_prime_tangled = 0;
    for (auto bfield_: bfields_) {
        local_b_prime = bfield_->bf_plasma_frame(point, v);
        b_prime += local_b_prime;
        b_prime_tangled += local_b_prime.norm()*bfield_->get_tangled_fraction(point);
    }

    //if(b_prime.norm() < eps_B || isnan(b_prime.norm())) {
    //    return std::make_tuple(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    //}
    // NaN means something is wrong!
    if(b_prime.norm() < eps_B) {
        return std::make_tuple(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    auto D = getD(n_los, v);
    auto nu_prime = nu/D;
    auto n_los_prime = get_n_los_prime(n_los, v);

    // Now calculate all coefficients
    double k_i_prime = 0.0;
    double k_q_prime = 0.0;
    double k_u_prime = 0.0;
    double k_v_prime = 0.0;
    double eta_i_prime = 0.0;
    double eta_q_prime = 0.0;
    double eta_u_prime = 0.0;
    double eta_v_prime = 0.0;
    double k_F_prime = 0.0;
    double k_C_prime = 0.0;
    double h_Q_prime = 0.0;

    double n_prime = 0.0;
    for(auto nfield_: nfields_) {
        n_prime = nfield_->nf_plasma_frame(point, gamma);
        k_i_prime += nfield_->particles_->k_i(b_prime, n_los_prime, nu_prime, n_prime);
        k_i_prime += nfield_->particles_->k_i(b_prime_tangled, n_los_prime, nu_prime, n_prime);
        k_q_prime += nfield_->particles_->k_q(b_prime, n_los_prime, nu_prime, n_prime);
        k_u_prime += nfield_->particles_->k_u(b_prime, n_los_prime, nu_prime, n_prime);
        k_v_prime += nfield_->particles_->k_v(b_prime, n_los_prime, nu_prime, n_prime);
        eta_i_prime += nfield_->particles_->eta_i(b_prime, n_los_prime, nu_prime, n_prime);
        eta_i_prime += nfield_->particles_->eta_i(b_prime_tangled, n_los_prime, nu_prime, n_prime);
        eta_q_prime += nfield_->particles_->eta_q(b_prime, n_los_prime, nu_prime, n_prime);
        eta_u_prime += nfield_->particles_->eta_u(b_prime, n_los_prime, nu_prime, n_prime);
        eta_v_prime += nfield_->particles_->eta_v(b_prime, n_los_prime, nu_prime, n_prime);
        k_F_prime += nfield_->particles_->k_F(b_prime, n_los_prime, nu_prime, n_prime);
        k_C_prime += nfield_->particles_->k_C(b_prime, n_los_prime, nu_prime, n_prime);
        h_Q_prime += nfield_->particles_->h_Q(b_prime, n_los_prime, nu_prime, n_prime);
    }

    //std::cout << "k_F = " << k_F_prime/D << std::endl;
    auto result = std::make_tuple(k_i_prime/D, k_q_prime/D, k_u_prime/D, k_v_prime/D,
                                  eta_i_prime*D*D, eta_q_prime*D*D, eta_u_prime*D*D, eta_v_prime*D*D,
                                  k_F_prime/D, k_C_prime/D, h_Q_prime/D);
    if(isnan(k_i_prime/D)) {
        std::cout << "NaN in k_I!" << std::endl;
    }
    return result;
}


std::tuple<double, double> Jet::get_stokes_I_transport_coefficients(Vector3d &point, Vector3d &n_los, double nu) {
    // Example for k_I (Lyutikov et al. 2005):
    // First, comoving frame ``k_i_prime`` (in the rest frame of the emission element) is connected to this ``k_i`` as
    // ``k_i = k_i_prime / D``. Second, in ``k_i_prime`` we need all quantities in comoving frame (primed) in terms of
    // lab frame:
    // b_prime = f(b, v)
    // n_los_prime = f(n_los, v)
    // nu_prime = f(nu, n_los, v) = nu/getD
    // n_prime = f(n, v) = n/Gamma

    Vector3d v = getV(point);
    auto gamma = getG(v);

    Vector3d b_prime{0.0, 0.0, 0.0};
    Vector3d local_b_prime{0.0, 0.0, 0.0};
    double b_prime_tangled = 0;
    for (auto bfield_: bfields_) {
        local_b_prime = bfield_->bf_plasma_frame(point, v);
        b_prime += local_b_prime;
        b_prime_tangled += local_b_prime.norm()*bfield_->get_tangled_fraction(point);
    }

//    std::cout << "B'[G] = " << b_prime.norm() << "\n";

    if(b_prime.norm() < eps_B) {
        return std::make_tuple(0.0, 0.0);
    }

    auto D = getD(n_los, v);
    auto nu_prime = nu/D;
    auto n_los_prime = get_n_los_prime(n_los, v);

    // Now calculate all coefficients
    double k_i_prime = 0.0;
    double eta_i_prime = 0.0;

    double n_prime = 0.0;
    for(auto nfield_: nfields_) {
        n_prime = nfield_->nf_plasma_frame(point, gamma);
//        std::cout << "N' = " << n_prime << "\n";
        k_i_prime += nfield_->particles_->k_i(b_prime, n_los_prime, nu_prime, n_prime);
        k_i_prime += nfield_->particles_->k_i(b_prime_tangled, n_los_prime, nu_prime, n_prime);
        eta_i_prime += nfield_->particles_->eta_i(b_prime, n_los_prime, nu_prime, n_prime);
        eta_i_prime += nfield_->particles_->eta_i(b_prime_tangled, n_los_prime, nu_prime, n_prime);
    }

//    std::cout << "=== k, eta = " << k_i_prime/D << ", " << eta_i_prime*D*D << "\n";

    return std::make_tuple(k_i_prime/D, eta_i_prime*D*D);
//    return std::make_tuple(1E+23*k_i_prime/D, 1e+23*eta_i_prime*D*D);
}




// This is k_i in lab frame that could be integrated along LOS.
double Jet::getKI(Vector3d &point, Vector3d &n_los, double nu) {
    // First, comoving frame ``k_i_prime`` (in the rest frame of the emission
    // element) is connected to this ``k_i`` as ``k_i = k_i_prime / D``.
    // Second, in ``k_i_prime`` we need all quantities in comoving frame
    // (primed) in terms of lab frame:
    // b_prime = f(b, v)
    // n_los_prime = f(n_los, v)
    // nu_prime = f(nu, n_los, v) = nu/getD
    // n_prime = f(n, v) = n/Gamma

    Vector3d v = getV(point);
//    std::cout << "V = " << v/c << "\n";
    auto gamma = getG(v);
//    std::cout << "Gamma = " << gamma << "\n";

    Vector3d b_prime{0.0, 0.0, 0.0};
    Vector3d local_b_prime{0.0, 0.0, 0.0};
    double b_prime_tangled = 0;
    for (auto bfield_: bfields_) {
        local_b_prime = bfield_->bf_plasma_frame(point, v);
        b_prime += local_b_prime;
        b_prime_tangled += local_b_prime.norm()*bfield_->get_tangled_fraction(point);
    }

//    std::cout << "b_prime[G] = " << b_prime.norm() << "\n";
//    std::cout << "b_prime_tangled = " << b_prime_tangled << "\n";

    //if(b_prime.norm() < eps_B || isnan(b_prime.norm())) {
    //    return 0.0;
    //}
    // NaN means something is wrong!
    if(b_prime.norm() < eps_B) {
//        std::cout << "B-field = " << b_prime.norm() << "\n";
        return 0.0;
    }

    auto D = getD(n_los, v);
//    std::cout << "D = " << D << "\n";
    auto nu_prime = nu/D;
    auto n_los_prime = get_n_los_prime(n_los, v);

    double k_i_prime = 0.0;

    double n_prime;
    for(auto nfield_: nfields_) {
        n_prime = nfield_->nf_plasma_frame(point, gamma);
//        std::cout << "N' = " << n_prime << "\n";
        k_i_prime += nfield_->particles_->k_i(b_prime, n_los_prime, nu_prime, n_prime);
        k_i_prime += nfield_->particles_->k_i(b_prime_tangled, n_los_prime, nu_prime, n_prime);
    }
    auto result = k_i_prime/D;
    if(isnan(result)) {
        std::cout << "NaN in k_I!" << std::endl;
    }
//    std::cout << "k_I = " << result << "\n";
    return result;
}

// This is eta_i in lab frame that could be integrated along LOS.
double Jet::getEtaI(Vector3d &point, Vector3d &n_los, double nu) {
    // First, comoving frame ``eta_i_prime`` (in the rest frame of the emission
    // element) is connected to this ``eta_i`` as ``eta_i = D^2 * eta_i_prime``.
    // Second, in ``eta_i_prime`` we need all quantities in comoving frame
    // (primed) in terms of lab frame:
    // b_prime = f(b, v)
    // n_los_prime = f(n_los, v)
    // nu_prime = f(nu, n_los, v) = nu/getD
    // n_prime = f(n, v) = n/Gamma

    Vector3d v = getV(point);
    auto gamma = getG(v);

    Vector3d b_prime{0.0, 0.0, 0.0};
    Vector3d local_b_prime{0.0, 0.0, 0.0};
    double b_prime_tangled = 0;
    for (auto bfield_: bfields_) {
        local_b_prime = bfield_->bf_plasma_frame(point, v);
        b_prime += local_b_prime;
        b_prime_tangled += local_b_prime.norm()*bfield_->get_tangled_fraction(point);
    }

    if(b_prime.norm() < eps_B || isnan(b_prime.norm())) {
        return 0.0;
    }

    auto D = getD(n_los, v);
    auto nu_prime = nu/D;
    auto n_los_prime = get_n_los_prime(n_los, v);

    double eta_i_prime = 0.0;
    double n_prime = 0.0;
    for(auto nfield_: nfields_) {
        n_prime = nfield_->nf_plasma_frame(point, gamma);
        eta_i_prime += nfield_->particles_->eta_i(b_prime, n_los_prime, nu_prime, n_prime);
        eta_i_prime += nfield_->particles_->eta_i(b_prime_tangled, n_los_prime, nu_prime, n_prime);
    }
    auto result = eta_i_prime*D*D;
    if(isnan(result)) {
        std::cout << "NaN in eta_I!" << std::endl;
    }
    return result;
}


double Jet::getKF(Vector3d &point, Vector3d &n_los, double nu) {

    Vector3d v = getV(point);
    auto gamma = getG(v);


    Vector3d b_prime{0.0, 0.0, 0.0};
    Vector3d local_b_prime{0.0, 0.0, 0.0};
    for (auto bfield_: bfields_) {
        local_b_prime = bfield_->bf_plasma_frame(point, v);
        b_prime += local_b_prime;
    }

    if(b_prime.norm() < eps_B || isnan(b_prime.norm())) {
        return 0.0;
    }

    auto D = getD(n_los, v);
    auto nu_prime = nu/D;
    auto n_los_prime = get_n_los_prime(n_los, v);

    double k_F_prime = 0.0;
    double n_prime = 0.0;
    for(auto nfield_: nfields_) {
        double n_prime = nfield_->nf_plasma_frame(point, gamma);
        k_F_prime += nfield_->particles_->k_F(b_prime, n_los_prime, nu_prime, n_prime);
    }
    auto result = k_F_prime/D;
    //std::cout << "k_F = " << result << std::endl;
    if(isnan(result)) {
        std::cout << "NaN in k_F!" << std::endl;
    }
    return result;
}


std::list<Intersection> Jet::hit(Ray &ray) {
    return geometry_->hit(ray);
}


Vector3d Jet::getV(const Vector3d &point) {
    auto v = vfield_->vf(point);
    if(v.norm() > c) {
        std::cout << "Speed > c!!!";
        throw PhysicalException("Speed");
    }
    return v;
}

//const Vector3d Jet::getB(const Vector3d &point) {
//    return bfield_->bf(point);
//}
//
//const Vector3d Jet::getBhat(const Vector3d& point) {
//    auto v = getV(point);
//    return bfield_->bhat_lab_frame(point, v);
//}

const Vector3d Jet::getB(const Vector3d &point) {
    auto v = getV(point);
    Vector3d b{0.0, 0.0, 0.0};
    for (auto bfield_: bfields_) {
        b += bfield_->bf_plasma_frame(point, v);
    }
    return b;
}

// FIXME: Is this linear operation?
const Vector3d Jet::getBhat(const Vector3d &point) {
    auto v = getV(point);
    Vector3d Bhat{0.0, 0.0, 0.0};
    for (auto bfield_: bfields_) {
        Bhat += bfield_->bhat_lab_frame(point, v);
    }
    return Bhat;
}