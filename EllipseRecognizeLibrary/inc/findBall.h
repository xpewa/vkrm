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
    int MIN_SIZE_OBJECT = 7500 / (scale*scale); // diameter = 100 (3 m)
    int MAX_SIZE_OBJECT = 750000 / (scale*scale); // diameter = 300 (1 m)
public:
    FindBall(ColorFilter const & colorFilter) : colorFilter(colorFilter) {}
    Ellipse findBall(cv::Mat const & img);
};

#endif //ELLIPSERECOGNIZE_FINDBALL_H
