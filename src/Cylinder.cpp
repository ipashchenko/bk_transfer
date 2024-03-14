#include <iostream>
#include <utils.h>
#include "Cylinder.h"

using std::min;
using std::max;


Cylinder::Cylinder(const Vector3d &origin, const Vector3d &direction, const double &r) {
    origin_ = origin;
    direction_ = direction;
    direction_.normalize();
    r_ = r;
}

const Vector3d& Cylinder::origin() const {
    return origin_;
}

const Vector3d& Cylinder::direction() const {
    return direction_;
}

const double Cylinder::r() const {
    return r_;
}

bool Cylinder::is_within(Vector3d& point) const {
    double distance = ((point - origin_).cross(direction_)).norm();
    return distance < r_;
}


std::list<Intersection> Cylinder::hit(Ray &ray) const {
    const Vector3d ray_origin = ray.origin()/pc;
    const Vector3d ray_direction = ray.direction();
    const Vector3d cilinder_origin = origin()/pc;
    const Vector3d cilinder_direction = direction();
    const Vector3d delta = ray_origin - cilinder_origin;
    const Vector3d v = ray_direction - ray_direction.dot(cilinder_direction) * cilinder_direction;
    double c2 = v.squaredNorm();
    const Vector3d w = delta - delta.dot(cilinder_direction) * cilinder_direction;
    double c1 = v.dot(w);
    double c0 = w.squaredNorm() - r()*r()/pc/pc;
    double d = c1*c1 - c0*c2;

    if (d < 0) {
//        std::cout << "No intersections!" << "\n";
        return std::list<Intersection>{};
    }
    else if (d > 0) {
        double eps = (c1 > 0 ? 1 : -1);
        double sqrt_d = sqrt(d);
        double t1 = (-c1 - eps*sqrt_d)/(c2);
        double t2 = c0/(-c1 - eps*sqrt_d);

        Vector3d point_in = ray.point(pc*min(t1, t2));
        Vector3d point_out = ray.point(pc*max(t1, t2));
//        std::cout << "TWO intersections!" << "\n";
//        std::cout << point_in/pc << ", " << point_out/pc << "\n";

        return std::list<Intersection>{Intersection(ray, point_in, point_out)};
    } else {
        // Single intersection
        if (c2 == 1 && c1 ==0 && c0 == 0) {
            double t = -c1/c2;
            Vector3d point = ray.point(t);
            return std::list<Intersection>{Intersection(ray, point, point)};
        }
        // Along border
        else if (c2 ==0 && c1 == 0 && c0 == 0) {
            return std::list<Intersection>{Intersection(ray, *this)};
        }
        // No intersection - along border outside
        else if (c2 == 0 && c1 == 0 && c0 > 0) {
            return std::list<Intersection>{};
        }
        // No intersection - along border inside
        else if (c2 == 0 && c1 == 0 && c0 <= 0) {
            return std::list<Intersection>{Intersection(ray, *this)};
        }
    }
}

const double Cylinder::big_scale() const {
  return 100.*r() ;
}

double Cylinder::radius_at_given_distance(const Vector3d &point) const {
    return r_;
}
