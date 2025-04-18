#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>

#include <filesystem>
namespace fs = std::filesystem;

#include "colorFilter.h"
#include "edgeDetection.h"
#include "detectEllipse.h"
#include "videoRecognize.h"
#include "findBall.h"

//std::string PATH = "../../img_color/IMG_";

std::string MODE = "IMAGE";
//std::string MODE = "IMAGE_SIMPLE";
//std::string MODE = "VIDEO";
std::string PATH = "../../Experiment1/Image_";
std::string PATH_IMAGE_TEST = "../../Experiment1/Image_45.bmp"; // 16, 45
//std::string PATH_IMAGE_TEST = "../../Experiment1/video1/Image_10.tiff";
//std::string PATH_VIDEO = "../../videos/video_3.MOV";
//std::string PATH_VIDEO = "../../Experiment1/video3_1/video_3_1.mp4";
std::string PATH_VIDEO = "../../Experiment1/video1/video_1_mp4/video1.mp4";
//std::string PATH_OUT_VIDEO = "../../videos/outVideo.mp4";
std::string PATH_OUT_VIDEO = "../../Experiment1/video1/video_1_mp4/out_video_1.mp4";
std::string _PATH_CYLINDER = "../../cylinder.txt";
int COUNT_FILER_IMG = 59;

void draw_ellipse(cv::Mat & img, Ellipse const & ellipse) {
    std::cout << "Ellipse center: (" << ellipse.x << ", " << ellipse.y << ")" << std::endl;

    cv::Point centerCircle1(ellipse.x, ellipse.y);
    cv::Scalar colorCircle1(0, 0, 255);
    cv::circle(img, centerCircle1, 10, colorCircle1, cv::FILLED);
    cv::imshow("img res", img);
    cv::waitKey(0);
}

int main(int argc, char const* argv[]) {

//     Color Filter
    ColorFilter colorFilter1;
    Cylinder cylinder = colorFilter1.train(PATH, PATH, ".bmp", ".png", COUNT_FILER_IMG);
    cylinder.save(_PATH_CYLINDER);
//    Cylinder cylinder;
    cylinder.load(_PATH_CYLINDER);
    ColorFilter colorFilter(cylinder);
    std::cout << "Cylinder R = " << cylinder.R << std::endl;
    std::cout << "Cylinder v = " << cylinder.v << std::endl;
    std::cout << "Cylinder p0 = " << cylinder.p0 << std::endl;
    std::cout << "Cylinder t1 = " << cylinder.t1 << std::endl;
    std::cout << "Cylinder t2 = " << cylinder.t2 << std::endl;

    if (MODE == "IMAGE") {
        cv::Mat img = cv::imread(PATH_IMAGE_TEST);
//    cv::Mat img = cv::imread("../../img_color/IMG_3.jpg");
//    cv::Mat img = cv::imread("../../image1.jpg");

//        int up_width = 1280 / 4;
//        int up_height = 1024 / 4;
        int up_width = 1000;
        int up_height = 800;
        resize(img, img, cv::Size(up_width, up_height), cv::INTER_LINEAR);

        cv::Mat img_new = colorFilter.recognize(img);
        cv::imshow("img", img);
        cv::waitKey(0);
        cv::imshow("colorFilter", img_new);
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

        // Get Ellipse parameters

        DetectEllipse detectEllipse;
        Ellipse ellipse = detectEllipse.detectEllipse(imagePoints);

        std::cout << "x: " << ellipse.x << std::endl;
        std::cout << "y: " << ellipse.y << std::endl;
        std::cout << "angle: " << ellipse.angle << std::endl;
        std::cout << "R1: " << ellipse.R1 << std::endl;
        std::cout << "R2: " << ellipse.R2 << std::endl;

        draw_ellipse(img, ellipse);
    }

    else if (MODE == "IMAGE_SIMPLE") {
        FindBall findBall(colorFilter);
        cv::Mat img = cv::imread(PATH_IMAGE_TEST);
        Ellipse ellipse = findBall.getEllipseParameters(img);
        cv::Mat img_clone = img.clone();
        draw_ellipse(img_clone, ellipse);

        Ball ball = findBall.findBall(img);
        std::cout << "x, y, z = " << ball.x << ", " << ball.y << ", " << ball.z << std::endl;
    }

    else if (MODE == "VIDEO") {
        cv::VideoCapture cap(PATH_VIDEO);
        if(!cap.isOpened()){
            std::cout << "Error opening video file" << std::endl;
            return -1;
        }
        VideoRecognize videoRecognize = VideoRecognize(colorFilter, cap);
        videoRecognize.recognize_ellipse_in_video(PATH_OUT_VIDEO);
    }

    return 0;
}
