#ifndef VKRM_VIDEORECOGNIZE_H
#define VKRM_VIDEORECOGNIZE_H

#include <opencv2/opencv.hpp>
#include "colorFilter.h"
#include "contous.h"
#include "edgeDetection.h"
#include "detectEllipse.h"


class VideoRecognize {
    ColorFilter& colorFilter;
    cv::VideoCapture cap;
    cv::VideoWriter video;

//    cv::Mat read_next_part_of_video();
public:
    VideoRecognize(ColorFilter& colorFilter, cv::VideoCapture cap) : cap(cap), colorFilter(colorFilter) {}
    ~VideoRecognize() { cap.release(); }

    void recognize_ellipse_in_video();
};

#endif //VKRM_VIDEORECOGNIZE_H
