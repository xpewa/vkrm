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


// Корреляция между изображениями.
double compareMasksWithMatchTemplate(const cv::Mat& mask1, const cv::Mat& mask2) {
    if (mask1.size() != mask2.size() || mask1.type() != mask2.type()) {
        throw std::runtime_error("Masks must have the same size and type");
    }
    cv::Mat result;
    cv::matchTemplate(mask1, mask2, result, cv::TM_CCOEFF_NORMED); // TM_CCOEFF_NORMED для нормализации

    double minVal, maxVal;
    cv::minMaxLoc(result, &minVal, &maxVal);
    return maxVal;
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


double calculatePrecision(const cv::Mat& mask1, const cv::Mat& mask2) {
    if (mask1.empty() || mask2.empty() || mask1.size() != mask2.size() || mask1.type() != CV_8U || mask2.type() != CV_8U) {
        throw std::runtime_error("Invalid input masks. Must be non-empty, same size, and type CV_8U.");
    }
    int tp = 0; // True Positives
    int fp = 0; // False Positives
    for (int y = 0; y < mask1.rows; ++y) {
        for (int x = 0; x < mask1.cols; ++x) {
            if (mask1.at<uchar>(y, x) == 255) {
                if (mask2.at<uchar>(y, x) == 255) {
                    tp++;
                }
                else {
                    fp++;
                }
            }

        }
    }
    if (tp + fp == 0) return 0.0;
    return static_cast<double>(tp) / (tp + fp);
}

double calculateRecall(const cv::Mat& mask1, const cv::Mat& mask2) {
    if (mask1.empty() || mask2.empty() || mask1.size() != mask2.size() || mask1.type() != CV_8U || mask2.type() != CV_8U) {
        throw std::runtime_error("Invalid input masks. Must be non-empty, same size, and type CV_8U.");
    }
    int tp = 0; // True Positives
    int fn = 0; // False Negatives
    for (int y = 0; y < mask1.rows; ++y) {
        for (int x = 0; x < mask1.cols; ++x) {
            if (mask2.at<uchar>(y, x) == 255) {
                if (mask1.at<uchar>(y, x) == 255) {
                    tp++;
                }
                else
                    fn++;
            }
        }
    }
    if (tp + fn == 0) return 0.0;
    return static_cast<double>(tp) / (tp + fn);
}

double calculateF1Score(const cv::Mat& mask1, const cv::Mat& mask2) {
    double precision = calculatePrecision(mask1, mask2);
    double recall = calculateRecall(mask1, mask2);
    if (precision + recall == 0) return 0.0;
    return 2.0 * (precision * recall) / (precision + recall);
}


void testColorFilter(std::string path_test_img, std::string path_mask_img, std::string type_img, int count_img) {
    ColorFilter colorFilter;
    Cylinder cylinder;

    cylinder.load(PATH_CYLINDER);
    if (cylinder.R) {
        colorFilter = ColorFilter(cylinder);
    }
    else {
        cylinder = colorFilter.train(PATH, PATH, ".bmp", ".png", count_img);
        cylinder.save(PATH_CYLINDER);
    }

    int mean_distance = 0;
    double mean_similarity_matchTemplate = 0;
    double mean_similarity = 0;
    double mean_precision = 0;
    double mean_recall = 0;
    double mean_f1_score = 0;
    try {
        for (int i = 1; i < count_img + 1; ++i) {
            cv::Mat img_test = cv::imread(path_test_img + std::to_string(i) + type_img);
            cv::Mat img_real = cv::imread(path_mask_img + std::to_string(i) + ".png",cv::IMREAD_GRAYSCALE);
            cv::Mat img_res = colorFilter.recognize(img_test);

//            cv::imshow("colorFilter", img_res);
//            cv::waitKey(0);

            mean_distance += hammingDistance(img_res, img_real);
            mean_similarity_matchTemplate += compareMasksWithMatchTemplate(img_res, img_real);
            mean_similarity += compareMasks(img_res, img_real);
            mean_precision += calculatePrecision(img_real, img_res);
            mean_recall += calculateRecall(img_real, img_res);
            mean_f1_score += calculateF1Score(img_real, img_res);
        }
        mean_distance /= count_img;
        mean_similarity_matchTemplate /= count_img;
        mean_similarity /= count_img;
        mean_precision /= count_img;
        mean_recall /= count_img;
        mean_f1_score /= count_img;

        std::cout << "Mean Hamming Distance: " << mean_distance << std::endl;
        std::cout << "Mean Similarity (matchTemplate): " << mean_similarity_matchTemplate * 100 << "%" << std::endl;
        std::cout << "Mean Similarity: " << mean_similarity * 100 << "%" << std::endl;
        std::cout << "Mean Precision: " << mean_precision << std::endl;
        std::cout << "Mean Recall: " << mean_recall << std::endl;
        std::cout << "Mean F1 Score: " << mean_f1_score << std::endl;
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

    std::string path_test_img_experiment_synthetic = "../../Experiment_synthetic/Image_";
    std::string path_mask_img_experiment_synthetic = "../../Experiment_synthetic/mask/Image_";
    std::string type_img_experiment_synthetic = ".png";
    int count_img_experiment_synthetic = 14;

    std::string path_test_img_experiment_synthetic_2 = "../../Experiment_synthetic_2/Image_";
    std::string path_mask_img_experiment_synthetic_2 = "../../Experiment_synthetic_2/mask/Image_";
    std::string type_img_experiment_synthetic_2 = ".png";
    int count_img_experiment_synthetic_2 = 230;

//    testColorFilter(path_test_img_experiment_1, path_mask_img_experiment_1, type_img_experiment_1, count_img_experiment_1);
//    testColorFilter(path_test_img_experiment_1_video_3, path_mask_img_experiment_1_video_3, type_img_experiment_1_video_3, count_img_experiment_1_video_3);
//    testColorFilter(path_test_img_experiment_1_video_1, path_mask_img_experiment_1_video_1, type_img_experiment_1_video_1, count_img_experiment_1_video_1);
//    testColorFilter(path_test_img_experiment_synthetic, path_mask_img_experiment_synthetic, type_img_experiment_synthetic, count_img_experiment_synthetic);
    testColorFilter(path_test_img_experiment_synthetic_2, path_mask_img_experiment_synthetic_2, type_img_experiment_synthetic_2, count_img_experiment_synthetic_2);

    return 0;
}