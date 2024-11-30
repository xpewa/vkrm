#include <iostream>
#include <opencv2/opencv.hpp>
#include <cassert>
#include <fstream>

#include "colorFilter.h"

std::string PATH = "../../Experiment1/Image_";
std::string PATH_CYLINDER = "../cylinder.txt";

std::string PATH_TEST_IMG = "../../Experiment1/Image_";

int COUNT_FILER_IMG = 59;


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
        cylinder = colorFilter.train(PATH, COUNT_FILER_IMG);
        cylinder.save(PATH_CYLINDER);
    }
    std::cout << "cylinder R: " << cylinder.R << std::endl;

    // Тест 1: Пустое изображение
//    cv::Mat emptyImage(500, 500, CV_8U, cv::Scalar(0));
//    cv::Mat resultEmpty = colorFilter.recognize(emptyImage);
//    cv::imshow("img new", resultEmpty);
//    cv::waitKey(0);
//    assert(resultEmpty.empty());

    int mean_distance = 0;
    double mean_similarity_matchTemplate = 0;
    double mean_similarity = 0;
    try {
        for (int i = 1; i < COUNT_FILER_IMG + 1; ++i) {
            cv::Mat img_test = cv::imread(PATH_TEST_IMG + std::to_string(i) + ".bmp");
            cv::Mat img_real = cv::imread(PATH_TEST_IMG + std::to_string(i) + ".png",cv::IMREAD_GRAYSCALE);
            cv::Mat img_res = colorFilter.recognize(img_test);

            mean_distance += hammingDistance(img_res, img_real);
            mean_similarity_matchTemplate += compareMasksWithMatchTemplate(img_res, img_real);
            mean_similarity += compareMasks(img_res, img_real);
        }
        mean_distance /= COUNT_FILER_IMG;
        mean_similarity_matchTemplate /= COUNT_FILER_IMG;
        mean_similarity /= COUNT_FILER_IMG;

        std::cout << "Mean Hamming Distance: " << mean_distance << std::endl;

        std::cout << "Mean Similarity (matchTemplate): " << mean_similarity_matchTemplate * 100 << "%" << std::endl;

        std::cout << "Mean Similarity: " << mean_similarity * 100 << "%" << std::endl;
    } catch (const std::runtime_error& error) {
        std::cerr << "Error: " << error.what() << std::endl;
    }
}


int main() {
    testColorFilter();
    return 0;
}