#include <Eigen/Eigen>
#include <memory>
#include <utility>
#include <cmath>
#include "ImagePlane.h"

using Eigen::Vector3d;
using std::pair;


ImagePlane::ImagePlane(pair<unsigned long int,unsigned long int> image_size, double pixel_size_start,
    double pixel_size_stop, double los_angle, bool jet_side) : image_(image_size, pixel_size_start, pixel_size_stop, jet_side),
                                                los_angle_(los_angle),
                                                image_size(image_size),
                                                jet_side_(jet_side) {
    direction_ = Vector3d{-sin(los_angle), 0, -cos(los_angle)};
    Vector3d scale(1., 1., 1./sin(los_angle));
    // Initialize rays
    vector<Pixel>& pixels = image_.getPixels();
    for (unsigned long int i = 0; i < image_size.first; ++i) {
        for (unsigned long int j = 0; j < image_size.second; ++j) {
            Pixel pxl = pixels[i * image_size.second + j];
            // Pixels of image has coordinates of observer image (rotated on angle
            // (pi/2-alpha) around y-axis of original xyz jet frame.
            Vector3d coordinate = pxl.getCoordinate();
            // Rays have coordinates in (yz)-plane of the jet.
            coordinate = scale.array()*coordinate.array();
            auto ij = std::make_pair(i, j);
            auto ray = Ray(coordinate, direction_);
            rays_.push_back(std::move(ray));
        }
    }
}

vector<Pixel> &ImagePlane::getPixels() {
    return image_.getPixels();
}

vector<Ray> &ImagePlane::getRays() {
    return rays_;
}

vector<vector<double>> ImagePlane::getImage(string value) {
    return image_.getImage(std::move(value));
}

vector<vector<double>> ImagePlane::getPixelSizes() {
    return image_.getPixelSizes();
}

Vector3d ImagePlane::get_direction() {
    return direction_;
}