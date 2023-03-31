#ifndef BK_TRANSFER_BFIELD_H
#define BK_TRANSFER_BFIELD_H

#include <Eigen/Eigen>
#include "utils.h"
#include "Geometry.h"
#include "VField.h"

using Eigen::Vector3d;


const double l_eps_B = 0.001*pc;

// B-field that has no preferred direction, e.g. random. Specified in plasma frame
// as it can't be transferred to the plasma frame if specified in a lab frame.
class ScalarBField {
    public:
        virtual double _bf(const Vector3d &point, double t = 0.0) const = 0 ;
        double bf(const Vector3d &point, double t = 0.0) const;
        // In plasma frame
        double bf_plasma_frame(const Vector3d &point, Vector3d &v, double t = 0.0) const;
    protected:
        ScalarBField(Geometry* geometry_out=nullptr, Geometry* geometry_in=nullptr, VField* vfield= nullptr, bool in_plasma_frame=true);
        Geometry* geometry_in_;
        Geometry* geometry_out_;
        VField* vfield_;
        bool in_plasma_frame_;
};

class BKScalarBField : public ScalarBField {
    public:
        BKScalarBField(double b_0, double m_b, Geometry* geometry_out=nullptr, Geometry* geometry_in=nullptr,
                       VField* vfield= nullptr, bool in_plasma_frame=true);
        double _bf(const Vector3d &point, double t = 0.0) const override;
    private:
        double b_0_;
        double m_b_;
};




class FlareBKScalarBField : public BKScalarBField {
    public:
        FlareBKScalarBField(double b_0, double m_b, double amp, double t_start, double width_pc, double theta_los, double z,
                            Geometry* geometry_out, Geometry* geometry_in = nullptr,
                            VField* vfield = nullptr, bool in_plasma_frame = true);
        double _bf(const Vector3d &point, double t = 0.0) const override;
    private:
        // Flare amplitude. To add flare to stationary BKNField with some ``n_0``, ``n_n``, use the same ``n_n`` but
        // define flare's ``n_0_fl`` as ``n_0*(1 + A_N)``, where ``A_N`` - fractional increase/decrease of particles in
        // flare
        double amp_;
        double t_start_;
        double width_pc_;
        // LOS angle for the jet axis
        double theta_los_;
        // redshift
        double z_;
};







// B-field with vector values, e.g. ordered component or ordered component with cells, with specified fraction of the
// completely tangled component. Vector component can be specified in any frame (plasma or lab). It is essential to
// specify field in the lab (BH) frame. Tangled component is specified only in plasma frame as some fraction of the
// vector component.
class VectorBField {
    public:
        virtual Vector3d _bf(const Vector3d &point, double t = 0.0) const = 0 ;
        Vector3d bf(const Vector3d &point, double t = 0.0) const ;
        // B-field in plasma (comoving) frame. Needed for calculation of transfer coefficients
        Vector3d bf_plasma_frame(const Vector3d &point, Vector3d &v, double t = 0.0) const;
        // Tangled B-field component in plasma (comoving) frame. Needed for calculation of transfer coefficients
        double bf_tangled_plasma_frame(const Vector3d &point, Vector3d &v, double t = 0.0) const;
        // Unit vector of B-field in laboratory (observer) frame. Needed for calculation of polarization swing.
        Vector3d bhat_lab_frame(const Vector3d &point, Vector3d &v, double t = 0.0) const;
        double get_tangled_fraction(const Vector3d &point, double t = 0.0) const;

    protected:
        VectorBField(bool in_plasma_frame, double tangled_fraction, Geometry* geometry_out=nullptr, Geometry* geometry_in=nullptr);
        bool in_plasma_frame_;
        double tangled_fraction_;
        Geometry* geometry_in_;
        Geometry* geometry_out_;

};



//class ConstCylinderBField : public VectorBField {
//    public:
//        ConstCylinderBField(double b_0, double n_b, bool in_plasma_frame, double tangled_fraction=0.0) ;
//        Vector3d bf(const Vector3d &point) const override ;
//    private:
//        double b_0_;
//        double n_b_;
//
//};


// B-Field like ``ConstCylinder`` that depends on z-coordinate only
class ConstCylinderBFieldZ : public VectorBField {
    public:
        ConstCylinderBFieldZ (double b_0, double n_b, bool in_plasma_frame, double tangled_fraction=0.0, Geometry* geometry_out= nullptr, Geometry* geometry_in= nullptr) ;
        Vector3d _bf(const Vector3d &point, double t = 0.0) const override ;
    private:
        double b_0_;
        double n_b_;

};



class RadialConicalBField : public VectorBField {
    public:
        RadialConicalBField(double b_0, double n_b, bool in_plasma_frame, double tangled_fraction=0.0, Geometry* geometry_out= nullptr, Geometry* geometry_in= nullptr);
        Vector3d _bf(const Vector3d &point, double t = 0.0) const override ;
    private:
        double b_0_;
        double n_b_;
};


class ToroidalBField : public VectorBField {
    public:
        ToroidalBField(double b_0, double n_b, bool in_plasma_frame, double tangled_fraction=0.0, Geometry* geometry_out= nullptr, Geometry* geometry_in= nullptr) ;
        Vector3d _bf(const Vector3d &point, double t = 0.0) const override ;
    private:
        double b_0_;
        double n_b_;
};


class HelicalCylinderBField : public VectorBField {
    public:
        HelicalCylinderBField(double b_0, double pitch_angle, bool in_plasma_frame, double tangled_fraction=0.0,
                              Geometry* geometry_out= nullptr, Geometry* geometry_in= nullptr) ;
        Vector3d _bf(const Vector3d &point, double t = 0.0) const override ;
    private:
        double b_0_;
        double pitch_angle_;
};


class HelicalConicalBField : public VectorBField {
    public:
        HelicalConicalBField(double b_0, double n_b, double pitch_angle, bool in_plasma_frame, double tangled_fraction=0.0, Geometry* geometry_out= nullptr, Geometry* geometry_in= nullptr) ;
        Vector3d _bf(const Vector3d &point, double t = 0.0) const override ;
    private:
        double b_0_;
        double n_b_;
        double pitch_angle_;
};


//class SpiralConicalBField : public VectorBField {
//    public:
//        SpiralConicalBField(double b_0, double pitch_angle, bool in_plasma_frame, double tangled_fraction=0.0) ;
//        Vector3d bf(const Vector3d &point) const override ;
//    private:
//        double b_0_;
//        double pitch_angle_;
//};
//
//class ForceFreeCylindricalBField : public VectorBField {
//    public:
//        ForceFreeCylindricalBField(double b_0, double mu, bool in_plasma_frame, double tangled_fraction=0.0) ;
//        Vector3d bf(const Vector3d &point) const override ;
//    private:
//        double b_0_;
//        double mu_;
//};


class ReversedPinchCylindricalBField : public VectorBField {
    public:
        ReversedPinchCylindricalBField(double b_0, Geometry* geometry_out, double tangled_fraction=0.0, Geometry* geometry_in= nullptr);
        Vector3d _bf(const Vector3d &point, double t = 0.0) const override ;
    private:
        double b_0_;
};


class ReversedPinchConicalBField : public VectorBField {
    public:
        ReversedPinchConicalBField(double b_0, double n_b, Geometry* geometry_out, double tangled_fraction=0.0, Geometry* geometry_in= nullptr);
        Vector3d _bf(const Vector3d &point, double t = 0.0) const override ;
    private:
        double b_0_;
        double n_b_;
};


#endif //BK_TRANSFER_BFIELD_H
