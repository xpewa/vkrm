#ifndef ELLIPSERECOGNIZE_FINDBALL_H
#define ELLIPSERECOGNIZE_FINDBALL_H

#include <iostream>
#include "edgeDetection.h"
#include "detectEllipse.h"
#include "colorFilter.h"
#include <opencv2/opencv.hpp>


class Camera {
public:
    double fx = 50 / 0.028125; // 6 / 0.0048
    double fy = 50 / 0.023438;
//    double fx = 2.77 ; // 6 / 0.0048
//    double fy = 2.77 ;
    double cx = 640.0;
    double cy = 512.0;
    double k1, k2, k3, p1, p2 = 0;
    double rotx = 1.1345; // радианы
    double roty = 0;
    double rotz = 0.8029;
    double transx = 2.0;
    double transy = -2.0;
    double transz = 1.0;

    std::vector<std::vector<double>> get_internal_parameters() const { return {{fx, 0.0, cx}, {0.0, fy, cy}, {0.0, 0.0, 1.0}}; }
    std::vector<double> get_distortion_coefficients() const { return {k1, k2, k3, p1, p2}; }
    cv::Mat get_rotation_vector() const { return cv::Mat_<double>(3, 1) << rotx, roty, rotz; }
    std::vector<std::vector<double>> get_rotation_vector3x3() const { return {{rotx, 0.0, 0.0}, {0.0, roty, 0.0}, {0.0, 0.0, rotz}}; }
    cv::Mat get_translation_vector() const { return cv::Mat_<double>(3, 1) << transx, transy, transz; }
    cv::Mat get_camera_matrix() const {return cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1; }
    cv::Mat get_distortion_coeff() const { return cv::Mat_<double>(1, 5) << k1, k2, k3, p1, p2; }
    cv::Mat get_world_to_camera_matrix() const {
        cv::Mat R;
        cv::Rodrigues(get_rotation_vector(), R);
        return cv::Mat_<double>(3, 4) <<
        R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), transx,
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), transy,
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), transz;

    }
};


class Ball {
public:
    double x, y, z;
};


class FindBall {
    double BALL_RADIUS = 0.24 / 2;
    ColorFilter colorFilter;
    const int scale = 4;
    int MIN_SIZE_OBJECT = 100 / (scale*scale); // diameter = 100 (3 m) 7500
    int MAX_SIZE_OBJECT = 100000000 / (scale*scale); // diameter = 300 (1 m) 750000
//    Camera camera;
public:
    FindBall(ColorFilter const & colorFilter) : colorFilter(colorFilter) {}
    Ball findBall(cv::Mat const & img);
    Ellipse getEllipseParameters(cv::Mat const & img);
    Ball estimate3dCoords(Ellipse ellipse, Camera camera);
};

#endif //ELLIPSERECOGNIZE_FINDBALL_H
