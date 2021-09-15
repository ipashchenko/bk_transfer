#ifndef BK_TRANSFER_PARABALOID_H
#define BK_TRANSFER_PARABALOID_H

#include <Eigen/Eigen>
#include "Geometry.h"

using Eigen::Vector3d;

class Parabaloid : public Geometry {
    public:

        Parabaloid(const Vector3d &origin_, const Vector3d &direction_,
                   const double &r0_, const double &big_scale_, bool with_counter_jet = false);

        std::list<Intersection> hit(Ray &ray) const override;
        const Vector3d &origin() const override;
        const Vector3d &direction() const;
        double r0() const;
        bool is_within(Vector3d &point) const override;
        const double big_scale() const override;
        double radius_at_given_distance(const Vector3d& point) const override;

    private:
        Vector3d origin_;
        Vector3d direction_;
        double r0_;
        double big_scale_;
        bool with_counter_jet_;
};

#endif //BK_TRANSFER_PARABALOID_H
