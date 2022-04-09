#ifndef BK_TRANSFER_SYSTEMS_H
#define BK_TRANSFER_SYSTEMS_H

#include <iostream>
#include "Jet.h"


typedef std::vector<double> state_type;

class System {
    public:
        System(Jet* newjet, Vector3d &newpoint_in, Vector3d &newray_direction, double newnu);

	protected:
        Jet* jet;
        // Starting point of integration. It is the closest point of jet for ``tau`` and the furthest for ``I``.
        Vector3d point_start;
        // Always direction to the observer.
        Vector3d ray_direction;
        double nu;
};

class Tau : public System {
    public:
        Tau(Jet* newjet, Vector3d &newpoint_in, Vector3d &newray_direction, double newnu) :
            System(newjet, newpoint_in,newray_direction, newnu) {};
        void operator() (const double &x, double &dxdt, double t);
};


class TauFR : public System {
    public:
        TauFR(Jet* newjet, Vector3d &newpoint_in, Vector3d &newray_direction, double newnu) :
            System(newjet, newpoint_in,newray_direction, newnu) {};
        void operator() (const double &x, double &dxdt, double t);
};


// point_start is actually furthest point (i.e. ``point_out`` in intersection).
class I : public System {
    public:
        I(Jet* newjet, Vector3d &newpoint_in, Vector3d &newray_direction, double newnu) :
            System(newjet, newpoint_in, newray_direction, newnu) {};

        void operator() (const double &x, double &dxdt, double t);
};


class Speed : public System {
    public:
        Speed(Jet* newjet, Vector3d &newpoint_in, Vector3d &newray_direction, double newnu) :
            System(newjet, newpoint_in, newray_direction, newnu) {};

        void operator() (const double &x, double &dxdt, double t);
};


class FullStokes : public System {
    public:
		FullStokes(Jet* newjet, Vector3d &newpoint_in, Vector3d &newray_direction, double newnu) :
                System(newjet, newpoint_in, newray_direction, newnu) {};

        void operator() (const state_type &x, state_type &dxdt, double t);
};

bool check_opt_depth(double tau_max, const double &x);


#endif //BK_TRANSFER_SYSTEMS_H
