#ifndef BK_TRANSFER_INTERCEPTION_H
#define BK_TRANSFER_INTERCEPTION_H

#include <Eigen/Eigen>
#include "Ray.h"
#include "Geometry.h"

using Eigen::Vector3d;
class Geometry;
class Ray;


// This class describes part of Ray that goes inside Geometry object. Absence of
// intersections is described by an empty list that is returned by Geometry.hit
// method. In case of [half]infinite intersections there's general rules of
// obtaining border points that don't depend on Geometry objects. Thus they are
// part of the Intersection class.
class Intersection {
    public:
        // Default ctor
        Intersection() = default;
        // Ctor for finite intersections
        Intersection(const Ray& ray, const Vector3d& point_in, const Vector3d& point_out);
        Intersection(const Ray& ray, const std::pair<Vector3d, Vector3d>& borders);
        // Ctor for full infinite intersections
        Intersection(const Ray& ray, const Geometry &geo);
        // Ctor for half infinite intersections
        Intersection(const Ray& ray, const Vector3d& point, const Geometry& geo);

        Vector3d direction() const;
        void set_direction(const Vector3d& direction);
        void set_borders(const Vector3d& point_in, const Vector3d& point_out);
        void set_point_in(const Vector3d& point);
        void set_point_out(const Vector3d& point);
        double length_pc() const;
        bool is_finite() const;
        const std::pair<Vector3d, Vector3d>& get_path() const;

    private:
        Vector3d direction_;
        bool is_finite_;
        std::pair<Vector3d, Vector3d> borders_;
        void init(const Ray& ray, const Vector3d& point_in, const Vector3d& point_out);
};

#endif //BK_TRANSFER_INTERCEPTION_H
