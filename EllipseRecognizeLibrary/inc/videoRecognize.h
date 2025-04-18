#ifndef VKRM_VIDEORECOGNIZE_H
#define VKRM_VIDEORECOGNIZE_H

#include <opencv2/opencv.hpp>
#include "colorFilter.h"
#include "findBall.h"


class VideoRecognize {
    ColorFilter& colorFilter;
    cv::VideoCapture cap;
    cv::VideoWriter video;
    FindBall findBall;
public:
    VideoRecognize(ColorFilter& colorFilter, cv::VideoCapture cap) : cap(cap), colorFilter(colorFilter), findBall(FindBall(colorFilter)) {}
    ~VideoRecognize() { cap.release(); }

    void recognize_ellipse_in_video(std::string path_out_video="../../videos/outVideo.mp4");
};

#endif //VKRM_VIDEORECOGNIZE_H
