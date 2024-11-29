#ifndef VKRM_DETECTELLIPSE_H
#define VKRM_DETECTELLIPSE_H

#include <iostream>
#include "edgeDetection.h"
#include <opencv2/opencv.hpp>

class Ellipse {
public:
    int x, y;
    double R1, R2, angle;
    float A, B, C, D, E;
};

class DetectEllipse {
    Ellipse __mnk(std::vector<Point> const& pts);
public:
    Ellipse detectEllipse(std::vector<Point> const& pts);
};

#endif //VKRM_DETECTELLIPSE_H
