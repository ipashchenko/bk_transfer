#include <tuple>
#include "Jet.h"
#include "MyExceptions.h"

using Eigen::Vector3d;


Jet::Jet(BaseGeometry *newgeo, VField *newvfield, std::vector<ScalarBField*> newsbFields,
         std::vector<VectorBField*> newbFields, std::vector<NField*> newnFields) :
         t_obs_(0.0) {
    geometry_ = newgeo;
    vfield_ = newvfield;
    sbfields_ = newsbFields;
    bfields_ = newbFields;
    nfields_ = newnFields;
}


std::tuple<double, double, double, double, double, double, double, double, double, double, double> Jet::get_transport_coefficients(Vector3d &point, Vector3d &n_los, double nu, double ltt_delay) {
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

    for (auto sbfield_: sbfields_) {
        b_prime_tangled += sbfield_->bf_plasma_frame(point, v, t_obs_ + ltt_delay);
    }

    double b_sq = hypot(b_prime_tangled, b_prime.squaredNorm());

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
        n_prime = nfield_->nf_plasma_frame(point, gamma, t_obs_ + ltt_delay);
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


std::tuple<double, double> Jet::get_stokes_I_transport_coefficients(Vector3d &point, Vector3d &n_los, double nu, double ltt_delay) {
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
    for (auto sbfield_: sbfields_) {
        b_prime_tangled += sbfield_->bf_plasma_frame(point, v, t_obs_ + ltt_delay);
    }

    if(b_prime.norm() < eps_B && b_prime_tangled < eps_B){
        return std::make_tuple(0.0, 0.0);
    }

    double b_sq = hypot(b_prime_tangled, b_prime.squaredNorm());

    auto D = getD(n_los, v);
    auto nu_prime = nu/D;
    auto n_los_prime = get_n_los_prime(n_los, v);

    // Now calculate all coefficients
    double k_i_prime = 0.0;
    double eta_i_prime = 0.0;

    double n_prime;
    for(auto nfield_: nfields_) {
        n_prime = nfield_->nf_plasma_frame(point, gamma, t_obs_ + ltt_delay);
        k_i_prime += nfield_->particles_->k_i(b_prime, n_los_prime, nu_prime, n_prime);
        k_i_prime += nfield_->particles_->k_i(b_prime_tangled, n_los_prime, nu_prime, n_prime);
        eta_i_prime += nfield_->particles_->eta_i(b_prime, n_los_prime, nu_prime, n_prime);
        eta_i_prime += nfield_->particles_->eta_i(b_prime_tangled, n_los_prime, nu_prime, n_prime);
    }

    return std::make_tuple(k_i_prime/D, eta_i_prime*D*D);
}


// This is k_i in lab frame that could be integrated along LOS.
double Jet::getKI(Vector3d &point, Vector3d &n_los, double nu, double ltt_delay) {
    // First, comoving frame ``k_i_prime`` (in the rest frame of the emission
    // element) is connected to this ``k_i`` as ``k_i = k_i_prime / D``.
    // Second, in ``k_i_prime`` we need all quantities in comoving frame
    // (primed) in terms of lab frame:
    // b_prime = f(b, v)
    // n_los_prime = f(n_los, v)
    // nu_prime = f(nu, n_los, v) = nu/getD
    // n_prime = f(n, v) = n/Gamma

    Vector3d v = getV(point);
    auto gamma = getG(v);
//    std::cout << "G = " << gamma << "\n";

    Vector3d b_prime{0.0, 0.0, 0.0};
    Vector3d local_b_prime{0.0, 0.0, 0.0};
    double b_prime_tangled = 0;
    for (auto bfield_: bfields_) {
        local_b_prime = bfield_->bf_plasma_frame(point, v);
        b_prime += local_b_prime;
        b_prime_tangled += local_b_prime.norm()*bfield_->get_tangled_fraction(point);
    }
    for (auto sbfield_: sbfields_) {
        b_prime_tangled += sbfield_->bf_plasma_frame(point, v, t_obs_ + ltt_delay);
    }

//    std::cout << "B' = " << b_prime << "\n";

    // FIXME: DEBUG
    if(b_prime.norm() < eps_B && b_prime_tangled < eps_B) {
        return 0.0;
    }

    double b_sq = hypot(b_prime_tangled, b_prime.squaredNorm());

    auto D = getD(n_los, v);
//    std::cout << "D = " << D << "\n";
    auto nu_prime = nu/D;
    auto n_los_prime = get_n_los_prime(n_los, v);

    double k_i_prime = 0.0;

    double n_prime;
    for(auto nfield_: nfields_) {
        n_prime = nfield_->nf_plasma_frame(point, gamma, t_obs_ + ltt_delay);
//        std::cout << "N_prime = " << n_prime << "\n";
        k_i_prime += nfield_->particles_->k_i(b_prime, n_los_prime, nu_prime, n_prime);
        k_i_prime += nfield_->particles_->k_i(b_prime_tangled, n_los_prime, nu_prime, n_prime);
    }
//    std::cout << "Done N_prime \n";
    auto result = k_i_prime/D;
//    std::cout << "k_I = " << result << "\n";
    if(isnan(result)) {
//        std::cout << "NaN in k_I!" << std::endl;
        throw NaNResult();
    }
    if(result < 0){
        throw NegativeKI();
    }
    return result;
}

// This is eta_i in lab frame that could be integrated along LOS.
double Jet::getEtaI(Vector3d &point, Vector3d &n_los, double nu, double ltt_delay) {
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
    for (auto sbfield_: sbfields_) {
        b_prime_tangled += sbfield_->bf_plasma_frame(point, v, t_obs_ + ltt_delay);
    }

    if(b_prime.norm() < eps_B && b_prime_tangled < eps_B) {
        return 0.0;
    }

    double b_sq = hypot(b_prime_tangled, b_prime.squaredNorm());

    auto D = getD(n_los, v);
    auto nu_prime = nu/D;
    auto n_los_prime = get_n_los_prime(n_los, v);

    double eta_i_prime = 0.0;
    double n_prime;
    for(auto nfield_: nfields_) {
        n_prime = nfield_->nf_plasma_frame(point, gamma, t_obs_ + ltt_delay);
        eta_i_prime += nfield_->particles_->eta_i(b_prime, n_los_prime, nu_prime, n_prime);
        eta_i_prime += nfield_->particles_->eta_i(b_prime_tangled, n_los_prime, nu_prime, n_prime);
    }
    auto result = eta_i_prime*D*D;
    if(isnan(result)) {
        throw NaNResult();
    }
    return result;
}


double Jet::getKF(Vector3d &point, Vector3d &n_los, double nu, double ltt_delay) {

    Vector3d v = getV(point);
    auto gamma = getG(v);


    Vector3d b_prime{0.0, 0.0, 0.0};
    Vector3d local_b_prime{0.0, 0.0, 0.0};
    for (auto bfield_: bfields_) {
        local_b_prime = bfield_->bf_plasma_frame(point, v);
        b_prime += local_b_prime;
    }

    if(b_prime.norm() < eps_B) {
        return 0.0;
    }

    auto D = getD(n_los, v);
    auto nu_prime = nu/D;
    auto n_los_prime = get_n_los_prime(n_los, v);

    double k_F_prime = 0.0;
    double n_prime;
    for(auto nfield_: nfields_) {
        // FIXME: Here different particles distribution results in different k_F values. Thus, we add k_F_prime.
        n_prime = nfield_->nf_plasma_frame(point, gamma, t_obs_ + ltt_delay);
        k_F_prime += nfield_->particles_->k_F(b_prime, n_los_prime, nu_prime, n_prime);
    }
    auto result = k_F_prime/D;
    if(isnan(result)) {
        throw NaNResult();
    }
    return result;
}


std::list<Intersection> Jet::hit(Ray &ray) {
    return geometry_->hit(ray);
}


Vector3d Jet::getV(const Vector3d &point, double t) {
    auto v = vfield_->vf(point, t);
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

const Vector3d Jet::getB(const Vector3d &point, double t) {
    auto v = getV(point, t);
    Vector3d b{0.0, 0.0, 0.0};
    for (auto bfield_: bfields_) {
        b += bfield_->bf_plasma_frame(point, v, t);
    }
    return b;
}

// FIXME: Is this linear operation?
const Vector3d Jet::getBhat(const Vector3d &point, double t) {
    auto v = getV(point, t);
    Vector3d Bhat{0.0, 0.0, 0.0};
    for (auto bfield_: bfields_) {
        Bhat += bfield_->bhat_lab_frame(point, v, t);
    }
    return Bhat;
}

void Jet::set_t_obs(double t_obs) {
    t_obs_ = t_obs;
}
