#ifndef VKRM_COLORFILTER_H
#define VKRM_COLORFILTER_H

#include <iostream>
#include <opencv2/opencv.hpp>

class Cylinder {
public:
    cv::Mat v, p0;
    float t1, t2, R;
};

class ColorFilter {
    cv::Mat __getArrayFromData(cv::Mat const& img, cv::Mat const& mask);
    Cylinder __train(cv::Mat const& pts);
    Cylinder __getCylinder(int countImg);
    cv::Mat __ransac(cv::Mat const& pts);
    double __calculateDistancePointPoint(cv::Vec3b const& p1, cv::Vec3b const& p2);
    double __calculateDistancePointLine(cv::Vec3b const& pl1, cv::Vec3b const& pl2, cv::Vec3b const& point);
public:
    cv::Mat recognize(int countDatasetImg, cv::Mat const& img);

};

#endif //VKRM_COLORFILTER_H
