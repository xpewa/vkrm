#include <iostream>
#include <opencv2/opencv.hpp>
#include <cassert>
#include <fstream>
#include <string>
#include <numeric>


#include "colorFilter.h"
#include "edgeDetection.h"
#include "detectEllipse.h"


std::string PATH = "../../Experiment1/Image_";
std::string PATH_CYLINDER = "../../cylinder.txt";


std::vector<Ellipse> readCentersFromFile(const std::string& filename) {
    std::vector<Ellipse> ellipses;
    std::ifstream file(filename);

    if (file.is_open()) {
        std::string line;
        while (getline(file, line)) {
            std::stringstream ss(line);
            Ellipse ellipse;
            ss >> ellipse.x >> ellipse.y;
            if (ss) {
                ellipses.push_back(ellipse);
            } else {
                std::cerr << "Error reading line: " << line << std::endl;
            }
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
    return ellipses;
}


double calculateError(const Ellipse & real, const Ellipse & predicted) {
    double dx = predicted.x - real.x;
    double dy = predicted.y - real.y;
    return std::sqrt(dx * dx + dy * dy);
}

void calculateMetrics(const std::vector<Ellipse>& real_coords, const std::vector<Ellipse>& predicted_coords) {
    std::vector<double> errors;

    for (int i = 0; i < real_coords.size(); ++i) {
        errors.push_back(calculateError(real_coords[i], predicted_coords[i]));
    }

    double mae = std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
    double squared_sum = std::accumulate(errors.begin(), errors.end(), 0.0,
                                         [](double sum, double err) { return sum + err * err; });
    double rmse = std::sqrt(squared_sum / errors.size());
    double max_error = *std::max_element(errors.begin(), errors.end());
    double min_error = *std::min_element(errors.begin(), errors.end());

//    for (int i = 0; i < errors.size(); ++i) {
//        std::cout << "error i = " << i << "  : " << errors[i] << std::endl;
//    }

    std::cout << "Средняя ошибка (MAE): " << mae << std::endl;
    std::cout << "Среднеквадратичная ошибка (RMSE): " << rmse << std::endl;
    std::cout << "Максимальная ошибка: " << max_error << std::endl;
    std::cout << "Минимальная ошибка: " << min_error << std::endl;
}


void testDetectEllipseExperiment(std::string path_ellipse_centers, std::string path_test_img, std::string type_img, int count_img) {

    std::vector<Ellipse> real_centers = readCentersFromFile(path_ellipse_centers);
    std::vector<Ellipse> predict_centers;

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

    for (int i = 1; i < count_img + 1; ++i) {
        cv::Mat img_test = cv::imread(path_test_img + std::to_string(i) + type_img);

        cv::Mat img_res = colorFilter.recognize(img_test);

        EdgeDetection edgeDetection;
        std::vector<Point> imagePoints = edgeDetection.find_points(img_res);

        cv::Mat emptyImg = cv::Mat::zeros(cv::Size(img_res.cols, img_res.rows),CV_8UC1);

        DetectEllipse detectEllipse;
        Ellipse ellipse = detectEllipse.detectEllipse(imagePoints);

        predict_centers.push_back(ellipse);
    }

    calculateMetrics(real_centers, predict_centers);
};


int main() {

    std::string path_ellipse_centers_experiment_1 = "../../Experiment1/ellipse_centers.txt";
    std::string path_test_img_experiment_1 = "../../Experiment1/Image_";
    std::string type_img_experiment_1 = ".bmp";
    int count_img_experiment_1 = 59;

    std::string path_ellipse_centers_experiment_1_video_3 = "../../Experiment1/video3/ellipse_centers_video_3.txt";
    std::string path_test_img_experiment_1_video_3 = "../../Experiment1/video3/Image_";
    std::string type_img_experiment_1_video_3 = ".tiff";
    int count_img_experiment_1_video_3 = 53;

    std::string path_ellipse_centers_experiment_1_video_3_1 = "../../Experiment1/video3_1/ellipse_centers_video_3_1.txt";
    std::string path_test_img_experiment_1_video_3_1 = "../../Experiment1/video3_1/Image_";
    std::string type_img_experiment_1_video_3_1 = ".tiff";
    int count_img_experiment_1_video_3_1 = 30;

//    testDetectEllipseExperiment(path_ellipse_centers_experiment_1,
//                                path_test_img_experiment_1,
//                                type_img_experiment_1,
//                                count_img_experiment_1);
//    testDetectEllipseExperiment(path_ellipse_centers_experiment_1_video_3,
//                                path_test_img_experiment_1_video_3,
//                                type_img_experiment_1_video_3,
//                                count_img_experiment_1_video_3);
    testDetectEllipseExperiment(path_ellipse_centers_experiment_1_video_3_1,
                                path_test_img_experiment_1_video_3_1,
                                type_img_experiment_1_video_3_1,
                                count_img_experiment_1_video_3_1);

    return 0;
}