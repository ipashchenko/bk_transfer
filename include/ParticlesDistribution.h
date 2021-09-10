#ifndef JETPOL_INCLUDE_PARTICLESDISTRIBUTION_H
#define JETPOL_INCLUDE_PARTICLESDISTRIBUTION_H

#include <Eigen/Eigen>

using Eigen::Vector3d;


class ParticlesDistribution {
    public:
        virtual double k_i(double b, Vector3d &n_los, double nu, double n) const = 0;
        // Absorption coefficient for given vector of magnetic field ``b``, unit LOS
        // vector ``n_los`` and others measured in emission frame
        virtual double k_i(Vector3d &b, Vector3d &n_los, double nu, double n) const = 0;
        virtual double eta_i(double b, Vector3d &n_los, double nu, double n) const = 0;
        // Emission coefficient for given vector of magnetic field ``b``, unit LOS
        // vector ``n_los`` and others measured in emission frame
        virtual double eta_i(Vector3d &b, Vector3d &n_los, double nu, double n) const = 0;
        virtual double k_q(Vector3d &b, Vector3d &n_los, double nu, double n) const = 0;
        virtual double k_u(Vector3d &b, Vector3d &n_los, double nu, double n) const = 0;
        virtual double k_v(Vector3d &b, Vector3d &n_los, double nu, double n) const = 0;
        virtual double k_F(Vector3d &b, Vector3d &n_los, double nu, double n) const = 0;
        virtual double k_C(Vector3d &b, Vector3d &n_los, double nu, double n) const = 0;
        virtual double h_Q(Vector3d &b, Vector3d &n_los, double nu, double n) const = 0;
        virtual double eta_q(Vector3d &b, Vector3d &n_los, double nu, double n) const = 0;
        virtual double eta_u(Vector3d &b, Vector3d &n_los, double nu, double n) const = 0;
        virtual double eta_v(Vector3d &b, Vector3d &n_los, double nu, double n) const = 0;

        double k_F_c(Vector3d &b, Vector3d &n_los, double nu, double n) const;
        double k_C_c(Vector3d &b, Vector3d &n_los, double nu, double n) const;
};


class PowerLaw : public ParticlesDistribution {
    public:
        PowerLaw(double s, double gamma_min, std::string plasma="normal", bool changing_s=false, double ds=0.0);
        double get_s(const Vector3d &point) const;
        double get_s(Vector3d &b, Vector3d &n_los) const;

        double k_i(double b, Vector3d &n_los, double nu, double n) const override;
        double k_i(Vector3d &b, Vector3d &n_los, double nu, double n) const override;
        double eta_i(double b, Vector3d &n_los, double nu, double n) const override;
        double eta_i(Vector3d &b, Vector3d &n_los, double nu, double n) const override;
        double k_q(Vector3d &b, Vector3d &n_los, double nu, double n) const override;
        double k_u(Vector3d &b, Vector3d &n_los, double nu, double n) const override;
        double k_v(Vector3d &b, Vector3d &n_los, double nu, double n) const override;
        double k_F(Vector3d &b, Vector3d &n_los, double nu, double n) const override;
        double k_C(Vector3d &b, Vector3d &n_los, double nu, double n) const override;
        double h_Q(Vector3d &b, Vector3d &n_los, double nu, double n) const override;
        double eta_q(Vector3d &b, Vector3d &n_los, double nu, double n) const override;
        double eta_u(Vector3d &b, Vector3d &n_los, double nu, double n) const override;
        double eta_v(Vector3d &b, Vector3d &n_los, double nu, double n) const override;
    private:
        std::string plasma_;
        double s_;
        double gamma_min_;
        bool changing_s_;
        double ds_;
        double factor_ki_;
        double factor_ki_rnd_;
        double factor_kv_;
        double factor_kf_;
        double factor_etai_;
        double factor_etai_rnd_;
        double factor_etav_;
};


class Thermal : public ParticlesDistribution {
    public:
        double k_i(double b, Vector3d &n_los, double nu, double n) const override;
        double k_i(Vector3d &b, Vector3d &n_los, double nu, double n) const override;
        double eta_i(double b, Vector3d &n_los, double nu, double n) const override;
        double eta_i(Vector3d &b, Vector3d &n_los, double nu, double n) const override;
        double k_q(Vector3d &b, Vector3d &n_los, double nu, double n) const override;
        double k_u(Vector3d &b, Vector3d &n_los, double nu, double n) const override;
        double k_v(Vector3d &b, Vector3d &n_los, double nu, double n) const override;
        double k_F(Vector3d &b, Vector3d &n_los, double nu, double n) const override;
        double k_C(Vector3d &b, Vector3d &n_los, double nu, double n) const override;
        double h_Q(Vector3d &b, Vector3d &n_los, double nu, double n) const override;
        double eta_q(Vector3d &b, Vector3d &n_los, double nu, double n) const override;
        double eta_u(Vector3d &b, Vector3d &n_los, double nu, double n) const override;
        double eta_v(Vector3d &b, Vector3d &n_los, double nu, double n) const override;
};


#endif //JETPOL_INCLUDE_PARTICLESDISTRIBUTION_H
