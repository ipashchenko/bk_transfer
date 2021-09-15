#include <math.h>
#include <utils.h>
#include "Parabaloid.h"
#include "Geometry.h"


Parabaloid::Parabaloid(const Vector3d &origin, const Vector3d &direction, const double &r0, const double &big_scale,
                       bool with_counter_jet) {
    origin_ = origin;
    direction_ = direction;
    direction_.normalize();
    r0_ = r0;
    big_scale_ = big_scale;
    with_counter_jet_ = with_counter_jet;
}

const Vector3d& Parabaloid::origin() const {
    return origin_;
}

const Vector3d& Parabaloid::direction() const {
    return direction_;
}

double Parabaloid::r0() const {
    return r0_;
}

const double Parabaloid::big_scale() const {
    return big_scale_;
}

bool Parabaloid::is_within(Vector3d& point) const {
    return std::hypot(point[0], point[1]) < r0_*sqrt(abs(point[2]));
}

std::list<Intersection> Parabaloid::hit(Ray &ray) const {

    std::list<Intersection> result;
    std::list<Intersection> result_cj;

    const Vector3d ray_origin = ray.origin();
    std::list<double> ts = intersection(ray.origin(), ray.direction(), 1., 1., 0., 0., 0., 0., 0., 0., -r0_*r0_/pc);

    // No intersection - most common case
    if (ts.empty()) {
        result = std::list<Intersection>{};

    }
    // Two intersections
    else if (ts.size() == 2) {
        double t1 = ts.front();
        double t2 = ts.back();
        Vector3d point_in = ray.point(std::min(t1, t2));
        Vector3d point_out = ray.point(std::max(t1, t2));
        result = std::list<Intersection>{Intersection(ray, point_in, point_out)};
    }
    // One intersection - ignore it for now
    else {
        result = std::list<Intersection>{};
    }

    if(with_counter_jet_){
        std::list<double> ts_cj = intersection(ray.origin(), ray.direction(), 1., 1., 0., 0., 0., 0., 0., 0., +r0_*r0_/pc);

        // No intersection - most common case
        if (ts_cj.empty()) {
            result_cj = std::list<Intersection>{};
        }
        // Two intersections
        else if (ts_cj.size() == 2) {
            double t1 = ts_cj.front();
            double t2 = ts_cj.back();
            Vector3d point_in = ray.point(std::min(t1, t2));
            Vector3d point_out = ray.point(std::max(t1, t2));
            result_cj = std::list<Intersection>{Intersection(ray, point_in, point_out)};
        }
        // One intersection - ignore it for now
        else {
            result_cj = std::list<Intersection>{};
        }
    }
    else {
        result_cj = std::list<Intersection>{};
    }
    result.insert(result.end(), result_cj.begin(), result_cj.end());
    return result;
}

double Parabaloid::radius_at_given_distance(const Vector3d &point) const {
    return r0_*pow(abs(point[2])/pc, 0.5);
}