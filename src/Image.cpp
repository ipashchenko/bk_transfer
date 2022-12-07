#include <string>
#include <numeric>
#include "Image.h"
#include "logspace.h"
#include "linspace.h"


using std::array;
using std::vector;
using std::string;
using Eigen::Vector3d;


vector<vector<double>> Image::getImage(const string& value) {
	vector<vector<double>> image;
	image.resize(image_size_.first);
	for (unsigned long int i = 0; i < image_size_.first; ++i) {
		image[i].resize(image_size_.second);
	}
    for (unsigned long int i = 0; i < image_size_.first; ++i) {
        for (unsigned long int j = 0; j < image_size_.second; ++j) {
            image[i][j] = pixels_[i*image_size_.second + j].getValue(value);
        }
    }
	return image;
}

Image::Image(pair<unsigned long int, unsigned long int> image_size, double lg_pixel_size_start_cm,
    double lg_pixel_size_stop_cm, bool jet_side):
        jet_side_(jet_side),
        image_size_(image_size),
        pixel_size_start_(lg_pixel_size_start_cm),
        pixel_size_stop_(lg_pixel_size_stop_cm),
        num_of_pixels_(image_size.first*image_size.second),
        pixel_sizes_(),
        pixel_center_coordinates_along_(),
        pixel_center_coordinates_across_(),
        pixels_() {

    // Create array of pixel sizes /////////////////////////////////////////////////////////////////////////////////////

    // Logarithmic - simple and t
    auto pixel_sizes_along = pyLogspace(lg_pixel_size_start_cm, lg_pixel_size_stop_cm, image_size.second);
    std::cout << "LOG pixel size" << "\n";

//    // sqrt - for KH
//    std::vector<double> pixel_sizes_along = linspace(1, int(image_size.second), int(image_size.second));
//    for(size_t i=0;i<pixel_sizes_along.size();i++){
//        pixel_sizes_along[i] = sqrt(pixel_sizes_along[i])*pow(10.0, lg_pixel_size_start_cm);
//    }
//    std::cout << "SQRT pixel size" << "\n";

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    pixel_sizes_.reserve(image_size.first);
    for (unsigned long int i=0; i < image_size.first; ++i) {
        pixel_sizes_.push_back(pixel_sizes_along);
    }

    // Create array of pixel center coordinates (n_across, n_along) for direction ALONG the jet
    std::vector<double> cumsum_along;
    cumsum_along.reserve(pixel_sizes_along.size());
    std::partial_sum(pixel_sizes_along.begin(), pixel_sizes_along.end(), cumsum_along.begin());
    for(unsigned long int i = 0; i<pixel_sizes_along.size(); ++i) {
        cumsum_along[i] = cumsum_along[i] - 0.5*pixel_sizes_along[i];
    }

    pixel_center_coordinates_along_.resize(image_size.first);
    for (unsigned long int i=0; i < image_size.first; ++i) {
        pixel_center_coordinates_along_[i].resize(image_size.second);
        for (unsigned long int j=0; j < image_size.second; ++j) {
            pixel_center_coordinates_along_[i][j] = cumsum_along[j];
        }
    }

    // Create array of pixel center coordinates (n_across, n_along) for direction ACROSS the jet
    std::vector<std::vector<double>> cumsum_across;
    // In each of the n_along arrays will be cumsums of the pixel coordinates in transverse direction
    cumsum_across.resize(image_size.second);
    std::vector<double> pixel_sizes_transverse(image_size.first/2, 1.0);

    // Going along the jet with larger and larger pixel sizes
    for (unsigned long int i=0; i < image_size.second; ++i) {
        // Array of the constant pixel sizes across  [j] the jet at given distance [i] from center
        for (unsigned long int j=0; j < image_size.first/2; ++j) {
            pixel_sizes_transverse.at(j) = pixel_sizes_along.at(i)*pixel_sizes_transverse.at(j);
        }
        cumsum_across.at(i).resize(image_size.first/2);
        std::partial_sum(pixel_sizes_transverse.begin(), pixel_sizes_transverse.end(), cumsum_across[i].begin());

        for(unsigned long int k = 0; k<pixel_sizes_transverse.size(); ++k) {
            cumsum_across.at(i).at(k) = cumsum_across.at(i).at(k) - 0.5*pixel_sizes_transverse.at(k);
        }

        // Get ready for next transverse slice
        for (unsigned long int j=0; j < image_size.first/2; ++j) {
            pixel_sizes_transverse.at(j) = 1.0;
        }
    }

    // Flip
    std::vector<std::vector<double> > trans_vec(image_size.first/2, std::vector<double>(image_size.second));
    for (unsigned long int i = 0; i < image_size.second; i++)
    {
        for (unsigned long int j = 0; j < image_size.first/2; j++)
        {
            trans_vec[j][i] = cumsum_across[i][j];
        }
    }

    // Negative coordinates
    std::vector<std::vector<double> > trans_vec_neg(image_size.first/2, std::vector<double>(image_size.second));
    for (unsigned long int i = 0; i < image_size.second; i++)
    {
        for (unsigned long int j = 0; j < image_size.first/2; j++)
        {
            trans_vec_neg[j][i] = -trans_vec[j][i];
        }
    }

    // Flip positive coordinates
    std::vector<std::vector<double> > trans_vec_flip(image_size.first/2, std::vector<double>(image_size.second));
    for (unsigned long int i = 0; i < image_size.second; i++)
    {
        for (unsigned long int j = 0; j < image_size.first/2; j++)
        {
            trans_vec_flip[j][i] = trans_vec[image_size.first/2-j-1][i];
        }
    }

    // Concatenate flipped positive coordinates with negative
    pixel_center_coordinates_across_ = trans_vec_flip;
    pixel_center_coordinates_across_.insert(pixel_center_coordinates_across_.end(), trans_vec_neg.begin(),
        trans_vec_neg.end());

    // Across
    for (unsigned long int i = 0; i < image_size_.first; ++i) {
        // Along
        for (unsigned long int j = 0; j < image_size_.second; ++j) {
            Vector3d coordinate = getCoordinate(i, j);
            auto ij = std::make_pair(i, j);
            // Find pixel size for current i, j
            auto pxl = Pixel(pixel_sizes_[i][j], coordinate, ij);
            pixels_.push_back(std::move(pxl));
        }
    }
}

Vector3d Image::getCoordinate(unsigned long int i, unsigned long int j) {
    if (jet_side_) {
        return Vector3d{0, pixel_center_coordinates_across_[i][j], pixel_center_coordinates_along_[i][j]};
    } else {
        return Vector3d{0, pixel_center_coordinates_across_[i][j], -pixel_center_coordinates_along_[i][j]};
    }
}

vector<Pixel> &Image::getPixels() {
  return pixels_;
}

vector<vector<double>> Image::getPixelSizes() {
    return pixel_sizes_;
}
