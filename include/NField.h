#ifndef JETPOL_NFIELD_H
#define JETPOL_NFIELD_H

#include <Eigen/Eigen>
#include "utils.h"
#include "Geometry.h"
#include "VField.h"
#include "ParticlesDistribution.h"

using Eigen::Vector3d;

const double l_eps_N = 0.001*pc;

class NField {
        friend class Jet;
    public:
        virtual double _nf(const Vector3d &point, double t = 0.0) const = 0;
        double nf(const Vector3d &point, double t = 0.0) const;
        double nf_plasma_frame(const Vector3d &point, double gamma, double t = 0.0) const;


    protected:
        explicit NField(bool in_plasma_frame,
                        ParticlesDistribution* particles,
                        Geometry* geometry_out = nullptr,
                        Geometry* geometry_in = nullptr,
                        VField* vfield = nullptr);
        bool in_plasma_frame_;
        ParticlesDistribution* particles_;

        // Inner border
        Geometry* geometry_in_;
        // Outer border
        Geometry* geometry_out_;
        // Velocity considered in non-stationary case
        VField* vfield_;
};


class BKNField: public NField {
    public:
        BKNField(double n_0, double n_n, ParticlesDistribution* particles, bool in_plasma_frame,
                 Geometry* geometry_out, Geometry* geometry_in = nullptr, VField* vfield = nullptr);
        double _nf(const Vector3d &point, const double t = 0.0) const override;
        void set_heating_profile(double r_mean, double r_width, double background_fraction = 0.0);
        void set_spiral(double phase_0, double lambda_0, double amp_0);
        int number_of_spirals() const {
            return lambdas_0_.size();
        }

    private:
        double n_0_;
        double n_n_;
        // Heating profile
        double r_mean_;
        double r_width_;
        double background_fraction_;
        bool is_profile_set_;
        // Spirals
//        std::vector<double> zs_0_;
        std::vector<double> phases_0_;
        std::vector<double> lambdas_0_;
        std::vector<double> amps_0_;
        // Radius of spiral at z0 (~0.9*r(z0))
//        double C_amp_;
        // Wavelength of spiral at z0 (~30*r(z0))
//        double C_lambda_;
        double k_;
        double spiral_width_frac_;
        bool is_spirals_present_;
};


class FlareBKNField : public NField {
    public:
        FlareBKNField(double n_0, double n_n, double t_start, double width_pc, ParticlesDistribution* particles, bool in_plasma_frame,
                      double theta_los, double z,
                      Geometry* geometry_out, Geometry* geometry_in = nullptr,
                      VField* vfield = nullptr);
        double _nf(const Vector3d &point, const double t = 0.0) const override;
    private:
        // Flare amplitude. To add flare to stationary BKNField with some ``n_0``, ``n_n``, use the same ``n_n`` but
        // define flare's ``n_0_fl`` as ``n_0*(1 + A_N)``, where ``A_N`` - fractional increase/decrease of particles in
        // flare
        double n_0_;
        double n_n_;
        double t_start_;
        double width_pc_;
        // LOS angle for the jet axis
        double theta_los_;
        // redshift
        double z_;
};



//class FFNField: public PowerLawNField {
//    public:
//        FFNField(double M_BH, double A,  double nu, double s, double gamma_min);
//        double _nf(const Vector3d& point) const override;
//    private:
//        double M_BH_;
//        // Parameters of the stream function
//        double A_;
//        double nu_;
//        // Reference position
//        double z1_;
//        // Width of the Gaussian ring - 5*r_g
//        double delta_;
//        // Radius where n peaks on z = +/-z_1 - free parameters [0, 100*r_g]
//        double Rp_;
//        // Number density of nonthermal electrons at (R, z) = (R-p, +/-z_1)
//        double n0_;
//};
//
//


#endif //BK_TRANSFER_NFIELD_H
