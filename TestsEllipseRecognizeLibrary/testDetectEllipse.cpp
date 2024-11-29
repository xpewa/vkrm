#include <iostream>
#include <opencv2/opencv.hpp>
#include <cassert>
#include <fstream>

#include "detectEllipse.h"


std::string PATH_CYLINDER = "../cylinder.txt";

std::string PATH_TEST_IMG_1 = "../image1.jpg";
std::string PATH_REAL_IMG_1 = "../image1.jpg";

int COUNT_FILER_IMG = 20;


void testDetectEllipse() {
//    cv::Mat img_test_1 = cv::imread(PATH_TEST_IMG_1);
//    cv::Mat img_real_1 = cv::imread(PATH_REAL_IMG_1);
//
//    DetectEllipse detectEllipse;
//    Ellipse ellipse = detectEllipse.detectEllipse(img_test_1);
//    std::cout << "Ellipse center: (" << ellipse.x << ", " << ellipse.y << ")" << std::endl;
//
//    cv::Point centerCircle1(ellipse.x, ellipse.y);
//    cv::Scalar colorCircle1(0, 0, 255);
//    cv::circle(img, centerCircle1, 10, colorCircle1, cv::FILLED);
//    cv::imshow("img res", img);
//    cv::waitKey(0);




};


int main() {

    return 0;
}