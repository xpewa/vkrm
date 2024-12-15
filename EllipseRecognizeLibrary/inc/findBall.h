#ifndef ELLIPSERECOGNIZE_FINDBALL_H
#define ELLIPSERECOGNIZE_FINDBALL_H

#include <iostream>
#include "edgeDetection.h"
#include "detectEllipse.h"
#include "colorFilter.h"
#include <opencv2/opencv.hpp>

class FindBall {
    ColorFilter colorFilter;
    const int scale = 4;
public:
    FindBall(ColorFilter const & colorFilter) : colorFilter(colorFilter) {}
    Ellipse findBall(cv::Mat const & img);
};

#endif //ELLIPSERECOGNIZE_FINDBALL_H
