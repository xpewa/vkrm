#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>

#include <filesystem>
namespace fs = std::filesystem;

#include "colorFilter.h"
#include "contous.h"
#include "edgeDetection.h"
#include "detectEllipse.h"

std::string PATH = "../img_color/";
int COUNT_FILER_IMG = 20;

int main(int argc, char const* argv[]) {

//     Color Filter
    ColorFilter colorFilter;
    cv::Mat img = cv::imread("../img_color/IMG_3.jpg");
//    cv::Mat img = cv::imread("../image1.jpg");

    int up_width = 512;
    int up_height = 683;
    resize(img, img, cv::Size(up_width, up_height), cv::INTER_LINEAR);

    cv::Mat img_new = colorFilter.recognize(COUNT_FILER_IMG, img);
    cv::imshow("img", img);
    cv::waitKey(0);
    cv::imshow("img new", img_new);
    cv::waitKey(0);

    // Edge Detection
//    std::vector<cv::Mat> images;
//    for (const auto & file : fs::directory_iterator(PATH)) {
//        cv::Mat img = cv::imread(file.path(), cv::IMREAD_GRAYSCALE);
//        images.push_back(img);
//    }
    EdgeDetection edgeDetection;
//    cv::Mat image = images[1];
    std::vector<Point> imagePoints = edgeDetection.find_points(img_new);
    cv::Mat emptyImg = cv::Mat::zeros(cv::Size(img_new.cols, img_new.rows),CV_8UC1);
    emptyImg = edgeDetection.draw_points(emptyImg, imagePoints);
    cv::imshow("edgeDetection", emptyImg);
    cv::waitKey(0);

    // Edge Fitting

    // Find Ellipse

    // Get Ellipse parameters

    DetectEllipse detectEllipse;
    Ellipse ellipse = detectEllipse.detectEllipse(imagePoints);
    std::cout << "Ellipse center: (" << ellipse.x << ", " << ellipse.y << ")" << std::endl;

    cv::Point centerCircle1(ellipse.x, ellipse.y);
    cv::Scalar colorCircle1(0, 0, 255);
    cv::circle(img, centerCircle1, 10, colorCircle1, cv::FILLED);
    cv::imshow("img res", img);
    cv::waitKey(0);

    return 0;
}
