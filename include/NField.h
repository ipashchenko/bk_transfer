#ifndef JETPOL_NFIELD_H
#define JETPOL_NFIELD_H

#include <Eigen/Eigen>
#include "utils.h"
#include "Geometry.h"
#include "VField.h"
#include "BField.h"
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
        NField(NField *pField);

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
        double _nf(const Vector3d &point, double t = 0.0) const override;
        void set_heating_profile(double amp_border, double r_mean, double r_width, double amp_axis, double r_width_axis, double background_fraction = 0.0);
        void set_background_fraction(double background_fraction);
        void set_spiral(double phase_0, double lambda_0, double amp_0);
        int number_of_spirals() const {
            return lambdas_0_.size();
        }

    private:
        double n_0_;
        double n_n_;
        // Heating profile
        double amp_border_;
        double r_mean_;
        double r_width_;
        double amp_axis_;
        double r_width_axis_;
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




// Here we need V-field to transfer B-field to plasma frame via B' = B_lab/Gamma. Here we assume dominating transverse
// component.
class EquipartitionBKNfield : public NField {
    public:
        EquipartitionBKNfield(ParticlesDistribution* particles, std::vector<ScalarBField*> bfields,
                              Geometry* geometry_out, Geometry* geometry_in = nullptr, VField* vfield = nullptr,
                              double t_start = 0.0, double fraction = 1.0);
        double _nf(const Vector3d &point, double t = 0.0) const override;
    private:
        double t_start_;
        std::vector<ScalarBField*> bfields_;
        double fraction_;
};



// Here we need V-field to transfer B-field to plasma frame via B' = B_lab/Gamma. Here we assume dominating transverse
// component.
class EquipartitionKHNfield : public NField {
    public:
        EquipartitionKHNfield(ParticlesDistribution* particles, BKScalarBField* bfield,
                              Geometry* geometry_out, Geometry* geometry_in = nullptr, VField* vfield = nullptr,
                              double omega = 0.0, double t_start = 0.0);
        double _nf(const Vector3d &point, double t = 0.0) const override;
        void set_background_fraction(double background_fraction);
        void set_spiral(double phase_0, double lambda_0, double amp_0);
        void set_spiral_width_frac(std::vector<double> frac);
        void set_spiral_scale(std::vector<double> scale);
        int number_of_spirals() const {
            return lambdas_0_.size();
        }
    private:
        double t_start_;
        std::vector<double> scale_;
        BKScalarBField* bfield_;
        double background_fraction_;
        // Spirals
        std::vector<double> phases_0_;
        std::vector<double> lambdas_0_;
        std::vector<double> amps_0_;
        double k_;
        std::vector<double> spiral_width_frac_;
        double omega_;
};


class FlareBKNField : public NField {
    public:
        // ``vfield`` argument accounts for possible difference between pattern and flow velocity. The last one is
        // specified in ``Jet`` ctor.
        FlareBKNField(NField* bkg_nfield, double amp, double t_start, double width_pc, VField* flare_pattern_vfield);
        double _nf(const Vector3d &point, double t = 0.0) const override;
    private:
        // Flare amplitude. To add flare to stationary BKNField with some ``n_0``, ``n_n``, use the same ``n_n`` but
        // define flare's ``n_0_fl`` as ``n_0*(1 + A_N)``, where ``A_N`` - fractional increase/decrease of particles in
        // flare
        double amp_;
        double t_start_;
        double width_pc_;
        NField* bkg_nfield_;
        VField* flare_pattern_vfield_;
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
