#ifndef MHD_TRANSFER_IMAGEPLANE_H
#define MHD_TRANSFER_IMAGEPLANE_H

#include <array>
#include <Eigen/Eigen>
#include <memory>
#include "Image.h"
#include "Ray.h"

using std::array;
using std::pair;
using Eigen::Vector3d;


class ImagePlane {
    public:
        ImagePlane(pair<unsigned long int,unsigned long int> image_size, double pixel_size_start, double pixel_size_stop,
                   double los_angle, bool jet_side);
        vector<Pixel>& getPixels();
        vector<Ray>& getRays();
        vector<vector<double>> getImage(string value);
        vector<vector<double>> getPixelSizes();
        const pair<unsigned long int,unsigned long int> image_size;
        Vector3d get_direction();

    private:
        bool jet_side_;
        Image image_;
        double los_angle_;
        Vector3d direction_;
        vector<Ray> rays_;
};


#endif //MHD_TRANSFER_IMAGEPLANE_H
