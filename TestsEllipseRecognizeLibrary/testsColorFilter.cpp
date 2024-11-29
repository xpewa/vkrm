#include <iostream>
#include <opencv2/opencv.hpp>
#include <cassert>
#include <fstream>

#include "colorFilter.h"


std::string PATH_CYLINDER = "../cylinder.txt";

std::string PATH_TEST_IMG_1 = "../image1.jpg";
std::string PATH_REAL_IMG_1 = "../image1.jpg";

int COUNT_FILER_IMG = 20;


int hammingDistance(const cv::Mat& mask1, const cv::Mat& mask2) {
    if (mask1.size() != mask2.size() || mask1.type() != mask2.type()) {
        throw std::runtime_error("Masks must have the same size and type");
    }
    int distance = cv::countNonZero(mask1 != mask2); // Количество отличающихся пикселей
    return distance;
}


// Корреляция (коэффициент сходства) между изображениями.
double compareMasksWithMatchTemplate(const cv::Mat& mask1, const cv::Mat& mask2) {
    if (mask1.size() != mask2.size() || mask1.type() != mask2.type()) {
        throw std::runtime_error("Masks must have the same size and type");
    }

    cv::Mat result;
    cv::matchTemplate(mask1, mask2, result, cv::TM_CCOEFF_NORMED); // TM_CCOEFF_NORMED для нормализации
    double minVal, maxVal;
    cv::minMaxLoc(result, &minVal, &maxVal);
    return maxVal; // Максимальное значение - мера сходства
}


double compareMasks(const cv::Mat& mask1, const cv::Mat& mask2) {
    if (mask1.size() != mask2.size() || mask1.type() != mask2.type()) {
        throw std::runtime_error("Masks must have the same size and type");
    }

    int totalPixels = mask1.rows * mask1.cols;
    int matchingPixels = 0;

    for (int i = 0; i < mask1.rows; ++i) {
        for (int j = 0; j < mask1.cols; ++j) {
            if (mask1.at<uchar>(i, j) == mask2.at<uchar>(i, j)) {
                matchingPixels++;
            }
        }
    }

    return static_cast<double>(matchingPixels) / totalPixels; // Процент совпадений
}


void testColorFilter() {
    ColorFilter colorFilter;
    Cylinder cylinder;

    cylinder.load(PATH_CYLINDER);
    if (cylinder.R) {
        colorFilter = ColorFilter(cylinder);
    }
    else {
        cylinder = colorFilter.train(COUNT_FILER_IMG);
        cylinder.save(PATH_CYLINDER);
    }
    std::cout << "cylinder R: " << cylinder.R << std::endl;

    // Тест 1: Пустое изображение
    cv::Mat emptyImage(500, 500, CV_8U, cv::Scalar(0));
    cv::Mat resultEmpty = colorFilter.recognize(emptyImage);
//    cv::imshow("img new", resultEmpty);
//    cv::waitKey(0);
//    assert(resultEmpty.empty());


    cv::Mat img_test_1 = cv::imread(PATH_TEST_IMG_1);
    cv::Mat img_real_1 = cv::imread(PATH_REAL_IMG_1);
    cv::Mat img_res_1 = colorFilter.recognize(img_test_1);

    try {
        int distance = hammingDistance(img_res_1, img_real_1);
        std::cout << "Hamming Distance: " << distance << std::endl;

        double similarity_matchTemplate = compareMasksWithMatchTemplate(img_res_1, img_real_1);
        std::cout << "Similarity (matchTemplate): " << similarity_matchTemplate * 100 << "%" << std::endl;

        double similarity = compareMasks(img_res_1, img_real_1);
        std::cout << "Similarity: " << similarity * 100 << "%" << std::endl;
    } catch (const std::runtime_error& error) {
        std::cerr << "Error: " << error.what() << std::endl;
    }
}


int main() {
    testColorFilter();
    return 0;
}