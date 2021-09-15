#include <math.h>
#include <iostream>
#include "Cone.h"
#include <vector>
#include <utils.h>

using std::min;
using std::max;
using Eigen::Matrix3d;


Cone::Cone(const Vector3d &neworigin, const Vector3d &newdirection, const double &newangle, const double &newscale) {
    origin_ = neworigin;
    direction_ = newdirection;
    direction_.normalize();
    angle_ = newangle;
    cos_angle_ = cos(angle_);
    big_scale_ = newscale;

}

const Vector3d& Cone::origin() const {
    return origin_;
}

const Vector3d& Cone::direction() const {
    return direction_;
}

const double Cone::angle() const {
    return angle_;
}

const double Cone::big_scale() const {
    return big_scale_;
}

bool Cone::is_within(Vector3d& point) const {
    Vector3d diff = point-origin_;
    diff.normalize();
    double cosp = direction_.dot(diff);
    return cosp > cos_angle_ || cosp < -cos_angle_;
}

std::list<Intersection> Cone::hit(Ray &ray) const {
	const Vector3d& ray_origin = ray.origin();
	const Vector3d& ray_direction = ray.direction();
	const Vector3d& cone_origin = origin();
	const Vector3d& cone_direction = direction();
	Matrix3d eye_matrix;
	eye_matrix << 1, 0, 0,
			0, 1, 0,
			0, 0, 1;
	double cone_angle = angle();
	// DP
	const Vector3d delta = ray_origin - cone_origin;
	// M
	Matrix3d M = cone_direction * cone_direction.transpose() -
			cos(cone_angle)*cos(cone_angle)*eye_matrix;
	double c2 = ray_direction.transpose() * M * ray_direction;
	double c1 = ray_direction.transpose() * M * delta;
	double c0 = delta.transpose() * M * delta;
	double d = c1*c1 - c0*c2;
	if (d < 0) {
		// No intersections
		return std::list<Intersection>{};
	}
	else if (d == 0) {
		// One intersection if ray goes through apex of cone or it parallels to any
		// of it's generating lines.
		double t = -c1/c2;
		Vector3d point = ray.point(t);
		// FIXME: Need to use some pixel size -specific, not arbitrary small numbers
		if ((point-origin_).isMuchSmallerThan(1E-06, 1E-06) && c1 == 0.0) {
			// One intersection and infinite path before/past
			auto borders = (std::pair<Vector3d, Vector3d> &&) full_infinite_path(ray);
			return std::list<Intersection>{Intersection(ray, borders)};
		} else {
			// One intersection at apex
			t = -c1/c2;
			point = ray.point(t);
			return std::list<Intersection>{Intersection(ray, point, point)};
		}
	} else {
		// Infinite intersections only if ray parallel to ray direction.
		double eps = (c1 > 0 ? 1 : -1);
		double t1 = (-c1 - eps*sqrt(c1*c1-c2*c0))/(c2);
		double t2 = c0/(-c1 - eps*sqrt(c1*c1-c2*c0));
		Vector3d point_in = ray.point(min(t1, t2));
		Vector3d point_out = ray.point(max(t1, t2));
		double cos_ray_cone = ray_direction.dot(cone_direction);
		// TODO: Case of ray parallel to cone direction is very rare - can safely ignore this
		if (abs(cos_ray_cone) != 1.) {
			// Two intersection - finite case
			return std::list<Intersection>{Intersection(ray, point_in, point_out)};
		} else {
			// Two intersection - two half-infinite cases
			return std::list<Intersection>{Intersection(ray, point_out, *this),
			                               Intersection(ray, point_in, *this)};
		}
	}
}

double Cone::radius_at_given_distance(const Vector3d &point) const {
    // Assuming Cone origin (0, 0, 0) and direction (0, 0, 1)
    return abs(point[2])*tan(angle_);
}