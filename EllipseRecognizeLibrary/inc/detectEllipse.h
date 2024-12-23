#ifndef VKRM_DETECTELLIPSE_H
#define VKRM_DETECTELLIPSE_H

#include <iostream>
#include "edgeDetection.h"
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

class Ellipse {
public:
    int x, y;
    double R1, R2, angle;
    double A, B, C, D, E, F;

    Ellipse() : x(0), y(0), R1(0.0), R2(0.0), angle(0.0), A(0.0), B(0.0), C(0.0), D(0.0), E(0.0), F(0.0) {}
};

class DetectEllipse {
    Ellipse __mnk(std::vector<Point> const& pts);
public:
    Ellipse detectEllipse(std::vector<Point> const& pts);
};

#endif //VKRM_DETECTELLIPSE_H
