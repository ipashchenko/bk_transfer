#ifndef BK_TRANSFER_JET_H
#define BK_TRANSFER_JET_H


#include "Geometry.h"
#include "VField.h"
#include "BField.h"
#include "NField.h"
#include "utils.h"


class Jet {
    public:
        Jet(BaseGeometry* geo, VField* vfield, std::vector<ScalarBField*> sbFields, std::vector<VectorBField*> bFields, std::vector<NField*> nFields);

        // Get all coefficients
        std::tuple<double, double, double, double, double, double, double, double, double, double, double> get_transport_coefficients(Vector3d &point, Vector3d &n_los, double nu);
        // Get only k & eta (for Stokes I)
        std::tuple<double, double> get_stokes_I_transport_coefficients(Vector3d &point, Vector3d &n_los, double nu);

        // Absorption coefficient in ``point`` of the jet in the observer (lab)
        // frame. ``n`` is unit LOS vector in the observer frame.
        double getKI(Vector3d &point, Vector3d &n_los, double nu);

        // Emission coefficient in ``point`` of the jet in the observer (lab)
        // frame. ``n`` is unit LOS vector in the observer frame.
        double getEtaI(Vector3d &point, Vector3d &n_los, double nu);

        double getKF(Vector3d &point, Vector3d &n_los, double nu);

        std::list<Intersection> hit(Ray& ray);

        // Vector of the bulk motion speed (in cm/s) in the lab frame at point ``point``.
        Vector3d getV(const Vector3d& point);

        // Vector component of the B-field in the plasma frame at point ``point``.
        const Vector3d getB(const Vector3d& point);

        // Unit vector of the Vector B-field component in the lab frame at point ``point``.
        const Vector3d getBhat(const Vector3d& point);

        void set_t_obs(double t_obs);

    private:
        BaseGeometry* geometry_;
        VField* vfield_;
        std::vector<VectorBField*> bfields_;
        std::vector<ScalarBField*> sbfields_;
        std::vector<NField*> nfields_;
        double t_obs_;
};


#endif //BK_TRANSFER_JET_H
