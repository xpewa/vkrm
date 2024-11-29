#ifndef VKRM_COLORFILTER_H
#define VKRM_COLORFILTER_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>


class Cylinder {
public:
    cv::Mat v, p0;
    float t1, t2, R;

    Cylinder() : t1(0.0), t2(0.0), R(0.0) {
        this->v = cv::Mat(1, 3, CV_32F);
        this->p0 = cv::Mat(1, 3, CV_32F);
    }

    void save(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) return;

        file.write(reinterpret_cast<const char*>(v.data), v.total()*v.elemSize());
        file.write(reinterpret_cast<const char*>(p0.data), p0.total()*p0.elemSize());
        file.write(reinterpret_cast<const char*>(&t1), sizeof(t1));
        file.write(reinterpret_cast<const char*>(&t2), sizeof(t2));
        file.write(reinterpret_cast<const char*>(&R), sizeof(R));
        file.close();
    }

    bool load(const std::string& filename) {
        std::cout << "I`m loading Cylinder" << std::endl;
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) return false;

        file.read(reinterpret_cast<char*>(v.data), v.total()*v.elemSize());
        file.read(reinterpret_cast<char*>(p0.data), p0.total()*p0.elemSize());
        file.read(reinterpret_cast<char*>(&t1), sizeof(t1));
        file.read(reinterpret_cast<char*>(&t2), sizeof(t2));
        file.read(reinterpret_cast<char*>(&R), sizeof(R));
        file.close();
        return true;
    }
};

class ColorFilter {
    Cylinder cylinder;

    cv::Mat __getArrayFromData(cv::Mat const& img, cv::Mat const& mask);
    Cylinder __getCylinder(cv::Mat const& pts);
    cv::Mat __ransac(cv::Mat const& pts);
    double __calculateDistancePointPoint(cv::Vec3b const& p1, cv::Vec3b const& p2);
    double __calculateDistancePointLine(cv::Vec3b const& pl1, cv::Vec3b const& pl2, cv::Vec3b const& point);
public:
    ColorFilter() {}
    ColorFilter(Cylinder cylinder) : cylinder(cylinder) {}
    Cylinder train(std::string path, int countImg);
    cv::Mat recognize(cv::Mat const& img);

};

#endif //VKRM_COLORFILTER_H
