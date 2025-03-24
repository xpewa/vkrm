#ifndef ELLIPSERECOGNIZE_FINDBALL_H
#define ELLIPSERECOGNIZE_FINDBALL_H

#include <iostream>
#include <random>
#include <opencv2/opencv.hpp>
#include "edgeDetection.h"
#include "detectEllipse.h"
#include "colorFilter.h"


struct Vec3 {
    double x, y, z;

    double dot(const Vec3& b) const {
        return x * b.x + y * b.y + z * b.z;
    }

    double length() const {
        return std::sqrt(x * x + y * y + z * z);
    }

    Vec3 normalize() const {
        double length = std::sqrt(x * x + y * y + z * z);
        Vec3 result;
        if (length > 1e-9) {
            result.x = x / length;
            result.y = y / length;
            result.z = z / length;
        } else {
            std::cout << "ERROR normalize vector: [" << x << ", " << y << ", " << z << "]" << std::endl;
            return {x, y, z};
        }
        return result;
    }

    Vec3 cross(const Vec3& b) const {
        Vec3 result;
        result.x = y * b.z - z * b.y;
        result.y = z * b.x - x * b.z;
        result.z = x * b.y - y * b.x;
        return result;
    }
};


class Camera {
    void computeCameraExtrinsics();
public:
    double focal_length_mm = 50.0;
    double sensor_size_mm = 36.0;
    int width = 1280;
    int height = 1024;

    double rotx = 0; // в градусах !
    double roty = 0;
    double rotz = 0;
    double transx = 0;
    double transy = 0;
    double transz = 0;

    double k1, k2, k3, p1, p2 = 0;

    double fx = (focal_length_mm / sensor_size_mm) * width;
    double fy = (focal_length_mm / sensor_size_mm) * height;
    double cx = width / 2;
    double cy = height / 2;

    double R[3][3];
    Vec3 T;

    Camera() {
        computeCameraExtrinsics();
    }

    std::vector<std::vector<double>> get_internal_parameters() const { return {{fx, 0.0, cx}, {0.0, fy, cy}, {0.0, 0.0, 1.0}}; }
    std::vector<double> get_distortion_coefficients() const { return {k1, k2, k3, p1, p2}; }

    Vec3 cameraToWorld(const Vec3 & Pc);
};


class Ball {
public:
    double x, y, z;
};


class FindBall {
    ColorFilter colorFilter;
    Camera camera;
    Ellipse ellipse;
    std::vector<Point> edgePoints;
    double BALL_RADIUS = 0.12; // метры 0.146 0.24
    const int scale = 4;
    int MIN_SIZE_OBJECT = 100 / (scale*scale); // diameter = 100 (3 m) 7500
    int MAX_SIZE_OBJECT = 100000000 / (scale*scale); // diameter = 300 (1 m) 750000

    Vec3 pixelToRay(double u, double v);
    std::vector<Point> get2DPointsFromEllipse(int numPoints);
    Vec3 getNormalForPlane(const std::vector<Vec3>& points);
    double computeConeAngle(const std::vector<Vec3>& edgeVectors, const Vec3& planeNormal);
    std::vector<Point> getContourPoints(const cv::Mat& image, int n);
    Vec3 calculateBallCenter13(const Vec3& planeNormal, double coneApexAngle);
    Vec3 calculateBallCenter2();
    Vec3 calculateBallCenter4(const std::vector<Point>& points);

    Ball estimate3dCoords_1();
    Ball estimate3dCoords_2();
    Ball estimate3dCoords_3();
    Ball estimate3dCoords_4();
public:
    FindBall(ColorFilter const& colorFilter) : colorFilter(colorFilter) {}
    Ball findBall(cv::Mat const& img);
    Ellipse getEllipseParameters(cv::Mat const& img);
};

#endif //ELLIPSERECOGNIZE_FINDBALL_H
