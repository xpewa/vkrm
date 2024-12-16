#include <iostream>
#include <opencv2/opencv.hpp>
#include <cassert>
#include <fstream>

#include "colorFilter.h"

std::string PATH = "../../Experiment1/Image_";
std::string PATH_CYLINDER = "../../cylinder.txt";


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


void testColorFilter(std::string path_test_img, std::string path_mask_img, std::string type_img, int count_img) {
    ColorFilter colorFilter;
    Cylinder cylinder;

    cylinder.load(PATH_CYLINDER);
    if (cylinder.R) {
        colorFilter = ColorFilter(cylinder);
    }
    else {
        cylinder = colorFilter.train(PATH, count_img);
        cylinder.save(PATH_CYLINDER);
    }

    int mean_distance = 0;
    double mean_similarity_matchTemplate = 0;
    double mean_similarity = 0;
    try {
        for (int i = 1; i < count_img + 1; ++i) {
            cv::Mat img_test = cv::imread(path_test_img + std::to_string(i) + type_img);
            cv::Mat img_real = cv::imread(path_mask_img + std::to_string(i) + ".png",cv::IMREAD_GRAYSCALE);
            cv::Mat img_res = colorFilter.recognize(img_test);

            mean_distance += hammingDistance(img_res, img_real);
            mean_similarity_matchTemplate += compareMasksWithMatchTemplate(img_res, img_real);
            mean_similarity += compareMasks(img_res, img_real);
        }
        mean_distance /= count_img;
        mean_similarity_matchTemplate /= count_img;
        mean_similarity /= count_img;

        std::cout << "Mean Hamming Distance: " << mean_distance << std::endl;

        std::cout << "Mean Similarity (matchTemplate): " << mean_similarity_matchTemplate * 100 << "%" << std::endl;

        std::cout << "Mean Similarity: " << mean_similarity * 100 << "%" << std::endl;
    } catch (const std::runtime_error& error) {
        std::cerr << "Error: " << error.what() << std::endl;
    }
}


int main() {
    std::string path_test_img_experiment_1 = "../../Experiment1/Image_";
    std::string path_mask_img_experiment_1 = "../../Experiment1/Image_";
    std::string type_img_experiment_1 = ".bmp";
    int count_img_experiment_1 = 59;

    std::string path_test_img_experiment_1_video_3 = "../../Experiment1/video3/Image_";
    std::string path_mask_img_experiment_1_video_3 = "../../Experiment1/video3_mask/Image_";
    std::string type_img_experiment_1_video_3 = ".tiff";
    int count_img_experiment_1_video_3 = 53;

    std::string path_test_img_experiment_1_video_1 = "../../Experiment1/video1/Image_";
    std::string path_mask_img_experiment_1_video_1 = "../../Experiment1/video1/video_1_mask/Image_";
    std::string type_img_experiment_1_video_1 = ".tiff";
    int count_img_experiment_1_video_1 = 18;

//    testColorFilter(path_test_img_experiment_1, path_mask_img_experiment_1, type_img_experiment_1, count_img_experiment_1);
//    testColorFilter(path_test_img_experiment_1_video_3, path_mask_img_experiment_1_video_3, type_img_experiment_1_video_3, count_img_experiment_1_video_3);
    testColorFilter(path_test_img_experiment_1_video_1, path_mask_img_experiment_1_video_1, type_img_experiment_1_video_1, count_img_experiment_1_video_1);

    return 0;
}