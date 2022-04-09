#include "System.h"
#include "utils.h"
#include "MyExceptions.h"
#include <cmath>
#include <tuple>


System::System(Jet *newjet,
               Vector3d &newpoint_in,
               Vector3d &newray_direction,
               double newnu) {
    jet = newjet;
    point_start = newpoint_in;
    ray_direction = newray_direction;
    nu = newnu;
}


void Tau::operator()(const double &x, double &dxdt, const double t) {
//    std::cout << "===========================" << "\n";
//    std::cout << "x = " << x << ", dxdt = " << dxdt << ", t = " << t << "\n";
//    if(x < 0){
//        std::cout << "Negative tau = " << x << "\n";
//        throw NegativeTau();
//    }
//    std::cout << "tau = " << x << "\n";
    Vector3d point = point_start - t * ray_direction;

    ///////////////////////////// LTT delay ////////////////////////////////////////////////////////////////////////////
    // Find LTT delay from this point to the plane containing (0, 0, 0) and normal to the LOS. Here N_plane vector is
    // directed to the observer.
    // ltt_delay is positive for points that lay closer to the observer than the plane
    double ltt_delay = distance_plane_point(point, {0.0, 0.0, 0.0}, ray_direction)/c;
    ltt_delay = 0.0;
//    double extra_jet_delay_pc = distance_plane_point(point, {0.0, 0.0, 0.0}, ray_direction)/pc;
//    if(extra_jet_delay_pc < 10.0) {
//        std::cout << "Delay due to extra jet distance [pc] = " << extra_jet_delay_pc << "\n";
//        std::cout << "LTTD (days) = " << ltt_delay / 24. / 60. / 60. << "\n";
//    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//    std::cout << "point = " << point/pc << "\n";
    dxdt = jet->getKI(point, ray_direction, nu, ltt_delay);
//    std::cout << "k_i = " << dxdt << "\n";
}


void TauFR::operator()(const double &x, double &dxdt, double t) {
    Vector3d point = point_start + t * ray_direction;
    ///////////////////////////// LTT delay ////////////////////////////////////////////////////////////////////////////
    // Find LTT delay from this point to the plane containing (0, 0, 0) and normal to the LOS
    double ltt_delay = distance_plane_point(point, {0.0, 0.0, 0.0}, ray_direction)/c;
    ltt_delay = 0.0;
    ///////////////////////////////////////////////////////
    dxdt = jet->getKF(point, ray_direction, nu, ltt_delay);
}


//void I::operator()(const double &x, double &dxdt, const double t) {
//    std::cout << "I = " << x << "\n";
//    Vector3d point = point_start + t * ray_direction;
//
//    // If negative Stokes I resulted from previous step, remember its value to compensate for
//    double compensate_negative_I = 0.0;
//    if(x < 0) {
//        compensate_negative_I = -x;
//    }
//
//    double k_i, eta_i;
//    std::tie(k_i, eta_i) = jet->get_stokes_I_transport_coefficients(point, ray_direction, nu);
//
//    dxdt = eta_i - k_i*(x+compensate_negative_I);
//
//    // FIXME: This is possibly wrong
//    // This adds to previous step Stokes I, so add value that compensating negative Stokes I from previous step
//    dxdt += compensate_negative_I;
//}


// FIXME: Test w/o compensating negative
void I::operator()(const double &x, double &dxdt, const double t) {
//    std::cout << "I = " << x << "\n";
    // Check that I is non-negative after previous step
//    if(x < 0){
//        throw NegativeI();
//    }

    Vector3d point = point_start + t * ray_direction;
    ///////////////////////////// LTT delay ////////////////////////////////////////////////////////////////////////////
    // Find LTT delay from this point to the plane containing (0, 0, 0) and normal to the LOS
    double ltt_delay = distance_plane_point(point, {0.0, 0.0, 0.0}, ray_direction)/c;
    ltt_delay = 0.0;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    double k_i, eta_i;
    std::tie(k_i, eta_i) = jet->get_stokes_I_transport_coefficients(point, ray_direction, nu, ltt_delay);

//    std::cout << "K_i = " << k_i << ", eta_i = " << eta_i << "\n";
    dxdt = eta_i - k_i*x;
}


void Speed::operator()(const double &x, double &dxdt, const double t) {
    Vector3d point = point_start + t * ray_direction;
    ///////////////////////////// LTT delay ////////////////////////////////////////////////////////////////////////////
    // Find LTT delay from this point to the plane containing (0, 0, 0) and normal to the LOS
    double ltt_delay = distance_plane_point(point, {0.0, 0.0, 0.0}, ray_direction)/c;
    ltt_delay = 0.0;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // If negative Stokes I resulted from previous step, remember its value to compensate for
    double compensate_negative_I = 0.0;
    if(x < 0) {
        compensate_negative_I = -x;
    }

    double k_i, eta_i;
    std::tie(k_i, eta_i) = jet->get_stokes_I_transport_coefficients(point, ray_direction, nu, ltt_delay);

    dxdt = eta_i - k_i*(x+compensate_negative_I);

    // This adds to previous step Stokes I, so add value that compensating negative Stokes I from previous step
    dxdt += compensate_negative_I;

    // Calculate apparent speed
    Vector3d beta = jet->getV(point)/c;
    // LOS angle is angle between the bulk motion speed and the local ``ray_direction`` (that goes from far side of the
    // source to the observer).
    double cosLOS = beta.dot(ray_direction)/beta.norm();
    double sinLOS = sqrt(1 - cosLOS*cosLOS);
//    std::cout << "cosLOS = " << cosLOS << ", sinLOS = " << sinLOS << "\n";
    double beta_app = beta.norm()*sinLOS / (1. - beta.norm()*cosLOS);
//    std::cout << "beta_app = " << beta_app << "\n";
    if(isnan(beta_app)) {
        beta_app = 0.0;
//        std::cout << "beta_app = NAN!" << "\n";
//        std::cout << "sinLOS = " << sinLOS << "\n";
//        std::cout << "beta = " << beta << "\n";
//        std::cout << "Psi = " << psi << "\n";
    }

    dxdt *= beta_app;
}


void FullStokes::operator()(const state_type &x, state_type &dxdt,
                            const double t) {
    Vector3d point = point_start + t * ray_direction;
    ///////////////////////////// LTT delay ////////////////////////////////////////////////////////////////////////////
    // Find LTT delay from this point to the plane containing (0, 0, 0) and normal to the LOS
    double ltt_delay = distance_plane_point(point, {0.0, 0.0, 0.0}, ray_direction)/c;
    ltt_delay = 0.0;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// If negative Stokes I resulted from previous step, remember its value to compensate for
	double compensate_negative_I = 0.0;
	if(x[0] < 0) {
	     compensate_negative_I = -x[0];
	}

	// TODO: Do I need this block of checking non-negligible B-field?
	// Find value of B in plasma frame to decide about making transport or not
    // TODO: Use retarded time!!!
	auto b_in_plasma_frame = jet->getB(point);
	auto bvalues_in_plasma_frame = b_in_plasma_frame.norm();

	// Non-zero B => make some transport!
	if(bvalues_in_plasma_frame > eps_B) {



        Vector3d B_hat = jet->getBhat(point);
        Vector3d beta = jet->getV(point)/c;
        // Unit vector of electric vector of wave (EVPA) in lab frame.
        // FIXME: If beta = 0 => e must be perpendicular to B_hat --- fixed inside ``e_hat`` (in function ``utils.q()``)
        Vector3d e = e_hat(B_hat, beta, ray_direction);
//        auto q_ = q(b_hat, v_beta, n_hat);
//        auto nq = n_hat.dot(q_);




        // Reference direction l (on sky along the jet axis). When chi = 0.5*atan2(U,Q) = 0 than EVPA should be along the
        // jet.
        Vector3d l_y(0.0, 1.0, 0.0);
        auto l = ray_direction.cross(l_y);

        // Need inverse rotation (on -2chi) from ``observers system`` to ``emitting system`` (see Porth's PhD)
        auto cos_chi = l.dot(e);
        auto sin_chi = ray_direction.dot(l.cross(e));
        auto sin_2chi = 2.0*sin_chi*cos_chi;
        auto cos_2chi = cos_chi*cos_chi - sin_chi*sin_chi;
//        std::cout << "cos2chi = " << cos_2chi << ", sin2chi = " << sin_2chi << ", s^2 + c^2 = " << sqrt(cos_2chi*cos_2chi + sin_2chi*sin_2chi) << "\n";
        // ``x_eb`` is State in ``emitting system``, ``x`` - in `observers system``
        auto x_eb = x;
        x_eb[1] = x[1]*cos_2chi - x[2]*sin_2chi;
        x_eb[2] = x[1]*sin_2chi + x[2]*cos_2chi;
        // Previous step compensated for negative Stokes I flux - to calculate change of state dxdt
        x_eb[0] += compensate_negative_I;

        // Change of state in ``emitting system``
        auto dxdt_eb = dxdt;

        double k_i;
        double k_q;
        double k_u;
        double k_v;
        double eta_i;
        double eta_q;
        double eta_u;
        double eta_v;
        double k_F;
        double k_C;
        double h_Q;

        std::tie(k_i, k_q, k_u, k_v, eta_i, eta_q, eta_u, eta_v, k_F, k_C, h_Q) = jet->get_transport_coefficients(point, ray_direction, nu, ltt_delay);

        dxdt_eb[0] = eta_i - k_i*x_eb[0] - k_q*x_eb[1] - k_u*x_eb[2] - k_v*x_eb[3];
        dxdt_eb[1] = eta_q - k_i*x_eb[1] - k_q*x_eb[0] - k_F*x_eb[2] - h_Q*x_eb[3];
        dxdt_eb[2] = eta_u - k_i*x_eb[2] - k_u*x_eb[0] + k_F*x_eb[1] - k_C*x_eb[3];
        dxdt_eb[3] = eta_v - k_i*x_eb[3] - k_v*x_eb[0] + h_Q*x_eb[1] + k_C*x_eb[2];

        // Need rotation (on 2chi) from ``emitting system` to ``observers system``
        // Change of state in ``observers system``
        dxdt = dxdt_eb;

        // FIXME: This is possibly wrong
        // This adds to previous step Stokes I, so add value that compensating negative Stokes I from previous step
        // FIXME: I must add to x[0] (however it is const)!!!
        dxdt[0] += compensate_negative_I;
        dxdt[1] = dxdt_eb[1]*cos_2chi + dxdt_eb[2]*sin_2chi;
        dxdt[2] = -dxdt_eb[1]*sin_2chi + dxdt_eb[2]*cos_2chi;

        if (isnan(dxdt.at(0)) || isnan(dxdt.at(1)) || isnan(dxdt.at(2)) || isnan(dxdt.at(3))) {
            std::cout.precision(15);
            std::cout << "Some Stokes are Nan during RT!" << std::endl;
            std::cout << "Current Stokes parameters =============== " << "\n";
            std::cout << "I = " << dxdt.at(0) << "\n";
            std::cout << "Q = " << dxdt.at(1) << "\n";
            std::cout << "U = " << dxdt.at(2) << "\n";
            std::cout << "V = " << dxdt.at(3) << "\n";
            std::cout << "Compensation of negative I = " << compensate_negative_I << "\n";
            std::cout << "Directions =============================== " << "\n";
            std::cout << "Ray direction : " << ray_direction << "\n";
            std::cout << "|Ray direction| : " << ray_direction.norm() << "\n";
            std::cout << "Reference direction on the sky l = " << l << "\n";
            std::cout << "|l| = " << l.norm() << "\n";
            std::cout << "Unit vector of electric vector of wave (EVPA) in lab frame e = " << e << "\n";
            std::cout << "|e| = " << e.norm() << "\n";
            std::cout << "Unit vector of B-field in the lab frame Bhat = " << B_hat << "\n";
            std::cout << "|Bhat| = " << B_hat.norm() << "\n";


            Vector3d B_plasma = jet->getB(point);
            std::cout << "Unit vector of B-field in the plasma frame Bplasma_hat = " << B_plasma.normalized() << "\n";

            std::cout << "Speed vector beta [c] = " << beta << "\n";
            std::cout << "|beta| = " << beta.norm() << "\n";
            auto q_debug = q(B_hat, beta, ray_direction);
            std::cout << "Vector q (in calculation of e): q = " << q_debug << "\n";
            auto nq_debug = ray_direction.dot(q_debug);
            std::cout << "nq (in calculation of e): nq = " << nq_debug << "\n";

            std::cout << "Ddenominator in e is : " << q_debug.squaredNorm() - nq_debug*nq_debug << "\n";

            std::cout << "CHECK n along q!" << "\n";
            std::cout << "|[ n x q ]| = " << (ray_direction.cross(q_debug)).norm() << "\n";

//            std::cout << "nx[vxBhat] = " <<

            std::cout << "Rotations ================================== " << "\n";
            std::cout << "cos(chi) = " << cos_chi << "\n";
            std::cout << "sin(chi) = " << sin_chi << "\n";
            std::cout << "cos(2chi) = " << cos_2chi << "\n";
            std::cout << "sin(2chi) = " << sin_2chi << "\n";
            std::cout << "Transport coefficients ============================ " << "\n";
            std::cout << "k_I = " << k_i << ", k_q = " << k_q << ",k_u = " << k_u << ", k_v = " << k_v <<
            ", eta_i = " << eta_i << ", eta_q = " << eta_q << ", eta_u = " << eta_u << ", eta_v = " << eta_v <<
            ", k_F = " << k_F << ", k_C = " << k_C << ", h_Q = " << h_Q << std::endl;
            std::cout << "Fields ============= " << "\n";
            std::cout << "B = " << bvalues_in_plasma_frame << "\n";
            throw NaNStokesComponents();
        }
    }

	// Zero B => no changes in radiation at all!
	else {
        dxdt[0] = 0.0;
        dxdt[1] = 0.0;
        dxdt[2] = 0.0;
        dxdt[3] = 0.0;
	}

}

bool check_opt_depth(double tau_max, const double &x) {
    return x >= tau_max;
}


//class push_back_state_and_time
//{
//    public:
//        push_back_state_and_time(std::vector<state_type> &states,
//                                 std::vector<double> &times,
//                                 std::vector<double> &P_0,
//                                 Jet* new_jet) :
//                counter(0),
//                m_states(states),
//                m_times(times),
//                m_P_cur(P_0)
//        {
//            jet = new_jet;
//        }
//        void operator()(const state_type &x, double t)
//        {
//            if(std::isnan(x[0])) {
//                throw JetBorderException();
//            }
//            //m_states.clear();
//            //m_times.clear();
//            m_states.push_back(x);
//            m_times.push_back(t);
//
//            double p_cur = solver->get_P0(x, t);
//            m_P_cur.push_back(p_cur);
//
//        }
//    private:
//        std::vector<state_type>& m_states;
//        std::vector<double>& m_times;
//        std::vector<double>& m_P_cur;
//        Jet* jet;
//        size_t counter;
//};
