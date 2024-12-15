#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>

#include <filesystem>
namespace fs = std::filesystem;

#include "colorFilter.h"
#include "contous.h"
#include "edgeDetection.h"
#include "detectEllipse.h"
#include "videoRecognize.h"

//std::string PATH = "../../img_color/IMG_";

std::string MODE = "IMAGE";
//std::string MODE = "VIDEO";
std::string PATH = "../../Experiment1/Image_";
//std::string PATH_IMAGE_TEST = "../../Experiment1/video1/Image_19.tiff"; // без мяча
std::string PATH_IMAGE_TEST = "../../Experiment1/Image_36.bmp"; // 16, 55
//std::string PATH_IMAGE_TEST = "../../Experiment1/video3/Image_20.tiff";
//std::string PATH_VIDEO = "../../videos/video_3.MOV";
std::string PATH_VIDEO = "../../Experiment1/video3_1/video_3_1.mp4";
//std::string PATH_OUT_VIDEO = "../../videos/outVideo.mp4";
std::string PATH_OUT_VIDEO = "../../Experiment1/video3_1/out_video_3_1.mp4";
std::string PATH_CYLINDER = "../../cylinder.txt";
int COUNT_FILER_IMG = 59;

int main(int argc, char const* argv[]) {

//     Color Filter
    ColorFilter colorFilter1;
    Cylinder cylinder = colorFilter1.train(PATH, COUNT_FILER_IMG);
    cylinder.save(PATH_CYLINDER);
//    Cylinder cylinder;
    cylinder.load(PATH_CYLINDER);
    ColorFilter colorFilter(cylinder);
    std::cout << "Cylinder R = " << cylinder.R << std::endl;

    if (MODE == "IMAGE") {
        cv::Mat img = cv::imread(PATH_IMAGE_TEST);
//    cv::Mat img = cv::imread("../../img_color/IMG_3.jpg");
//    cv::Mat img = cv::imread("../../image1.jpg");

        int up_width = 1000;
        int up_height = 800;
        resize(img, img, cv::Size(up_width, up_height), cv::INTER_LINEAR);

        cv::Mat img_new = colorFilter.recognize(img);
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

        // Get Ellipse parameters

        DetectEllipse detectEllipse;
        Ellipse ellipse = detectEllipse.detectEllipse(imagePoints);
        std::cout << "Ellipse center: (" << ellipse.x << ", " << ellipse.y << ")" << std::endl;

        cv::Point centerCircle1(ellipse.x, ellipse.y);
        cv::Scalar colorCircle1(0, 0, 255);
        cv::circle(img, centerCircle1, 10, colorCircle1, cv::FILLED);
        cv::imshow("img res", img);
        cv::waitKey(0);
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
