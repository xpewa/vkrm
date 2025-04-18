#include "colorFilter.h"
#include "edgeDetection.h"
#include "detectEllipse.h"
#include "findBall.h"

#include "testCommon.h"


void showHistogram(const std::vector<double>& data, const std::string& windowName, int numBins = 10) {
    if (data.empty()) {
        std::cout << "No data to display." << std::endl;
        return;
    }
    int minVal = *std::min_element(data.begin(), data.end());
    int maxVal = *std::max_element(data.begin(), data.end());
    double binSize = static_cast<double>(maxVal - minVal) / numBins;
    std::vector<int> bins(numBins, 0);
    for (int val : data) {
        int binIndex = static_cast<int>((val - minVal) / binSize);
        if (binIndex == numBins)
            binIndex--;
        bins[binIndex]++;
    }
    int histHeight = 800;
    int histWidth = 1200;
    int textMargin = 20;
    cv::Mat histImage(histHeight + 2 * textMargin, histWidth + 2* textMargin, CV_8UC3, cv::Scalar(255, 255, 255));
    int maxBinCount = *std::max_element(bins.begin(), bins.end());
    if (maxBinCount == 0) maxBinCount = 1;
    int binWidth = histWidth / numBins;
    int endY = histHeight + textMargin;
    for (int i = 0; i < numBins; ++i) {
        int barHeight = static_cast<int>(static_cast<double>(bins[i]) / maxBinCount * histHeight);
        cv::rectangle(histImage, cv::Point(textMargin + i * binWidth, endY),
                      cv::Point(textMargin + (i + 1) * binWidth, endY - barHeight),
                      cv::Scalar(123, 104, 238), -1);
    }
    for (int i = 0; i < numBins; ++i) {
        std::string label = std::to_string((int) (minVal + i * binSize));
        int x = textMargin + i * binWidth;
        int y = histHeight + textMargin + textMargin/2;
        cv::putText(histImage, label, cv::Point(x,y), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
    }
    cv::imshow(windowName, histImage);
    cv::waitKey(0);
}

double calculateError(const Ellipse & real, const Ellipse & predicted) {
    double dx = predicted.x - real.x;
    double dy = predicted.y - real.y;
    return std::sqrt(dx * dx + dy * dy);
}

void calculateMetrics(const std::vector<Ellipse>& real_coords, const std::vector<Ellipse>& predicted_coords) {
    std::vector<double> errors;

    for (int i = 0; i < predicted_coords.size(); ++i) {
        errors.push_back(calculateError(real_coords[i], predicted_coords[i]));
    }

    double mae = std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
    double squared_sum = std::accumulate(errors.begin(), errors.end(), 0.0,
                                         [](double sum, double err) { return sum + err * err; });
    double rmse = std::sqrt(squared_sum / errors.size());
    double max_error = *std::max_element(errors.begin(), errors.end());
    double min_error = *std::min_element(errors.begin(), errors.end());

//    showHistogram(errors, "Error Histogram", 12);

    std::cout << "Средняя ошибка (MAE): " << mae << std::endl;
    std::cout << "Среднеквадратичная ошибка (RMSE): " << rmse << std::endl;
    std::cout << "Максимальная ошибка: " << max_error << std::endl;
    std::cout << "Минимальная ошибка: " << min_error << std::endl;
}


void testDetectEllipseExperiment(const std::string& path_ellipse_centers, const std::string& path_test_img, const std::string& type_img, int count_img,
                                 const std::string& path_to_data_for_cylinder, const std::string& path_to_mask_for_cylinder, const std::string& type_img_for_cylinder, const std::string& type_mask_for_cylinder) {

    std::vector<Ellipse> real_centers = readCentersFromFile(path_ellipse_centers);
    std::vector<Ellipse> predict_centers;

    ColorFilter colorFilter;
    Cylinder cylinder;
    cylinder.load(PATH_CYLINDER);
    if (cylinder.R) {
        colorFilter = ColorFilter(cylinder);
    }
    else {
        cylinder = colorFilter.train(path_to_data_for_cylinder, path_to_mask_for_cylinder, type_img_for_cylinder, type_mask_for_cylinder, count_img);
        cylinder.save(PATH_CYLINDER);
    }
    FindBall findBall(colorFilter);

    std::vector<long long> times;

    for (int i = 1; i < count_img + 1; ++i) {

//        auto start = std::chrono::high_resolution_clock::now();
//
        cv::Mat img_test = cv::imread(path_test_img + std::to_string(i) + type_img);
//
//        cv::Mat img_res = colorFilter.recognize(img_test);
//
//        EdgeDetection edgeDetection;
//        std::vector<Point> imagePoints = edgeDetection.find_points(img_res);
//        cv::Mat img_res_det = cv::Mat::zeros(cv::Size(img_res.cols, img_res.rows),CV_8UC1);
//        img_res_det = edgeDetection.draw_points(img_res_det, imagePoints);
//        cv::imshow("img_res", img_res);
//        cv::waitKey(0);
//        cv::imshow("img_res_det", img_res_det);
//        cv::waitKey(0);
//
//        DetectEllipse detectEllipse;
//        Ellipse ellipse = detectEllipse.detectEllipse(imagePoints);

//        auto end = std::chrono::high_resolution_clock::now();

        auto start = std::chrono::high_resolution_clock::now();

        Ellipse ellipse = findBall.getEllipseParameters(img_test);

        auto end = std::chrono::high_resolution_clock::now();

        predict_centers.push_back(ellipse);

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        times.push_back(duration.count());
    }

    double averageTime = accumulate(times.begin(), times.end(), 0.0) / times.size();
    std::cout << "Среднее время обработки изображения: " << averageTime << " microseconds" << std::endl;
//    for (int i = 0; i < times.size(); ++i) {
//        std::cout << "times " << i << ": " << times[i] << std::endl;
//    }

    calculateMetrics(real_centers, predict_centers);
};


int main() {

    std::string path_data = "../../Experiment1/Image_";

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

    std::string path_ellipse_centers_experiment_1_video_1 = "../../Experiment1/video1/ellipse_centers_video_1.txt";
    std::string path_test_img_experiment_1_video_1 = "../../Experiment1/video1/Image_";
    std::string type_img_experiment_1_video_1 = ".tiff";
    int count_img_experiment_1_video_1 = 18;

    std::string path_data_img_synthetic = "../../Experiment_synthetic_2/Image_";
    std::string path_data_mask_synthetic = "../../Experiment_synthetic_2/mask/Image_";

    std::string path_ellipse_centers_experiment_synthetic = "../../Experiment_synthetic_2/ball_center.txt";
    std::string path_test_img_experiment_synthetic = "../../Experiment_synthetic_2/Image_";
    std::string type_img_experiment_synthetic = ".png";
    int count_img_experiment_synthetic = 660;

//    testDetectEllipseExperiment(path_ellipse_centers_experiment_1,
//                                path_test_img_experiment_1,
//                                type_img_experiment_1,
//                                count_img_experiment_1,
//                                path_data,
//                                path_data,
//                                type_img_experiment_1,
//                                ".png");
//    testDetectEllipseExperiment(path_ellipse_centers_experiment_1_video_3,
//                                path_test_img_experiment_1_video_3,
//                                type_img_experiment_1_video_3,
//                                count_img_experiment_1_video_3,
//                                path_data,
//                                path_data,
//                                type_img_experiment_1,
//                                ".png");
//    testDetectEllipseExperiment(path_ellipse_centers_experiment_1_video_3_1,
//                                path_test_img_experiment_1_video_3_1,
//                                type_img_experiment_1_video_3_1,
//                                count_img_experiment_1_video_3_1,
//                                path_data,
//                                path_data,
//                                type_img_experiment_1,
//                                ".png");
//    testDetectEllipseExperiment(path_ellipse_centers_experiment_1_video_1,
//                                path_test_img_experiment_1_video_1,
//                                type_img_experiment_1_video_1,
//                                count_img_experiment_1_video_1,
//                                path_data,
//                                path_data,
//                                type_img_experiment_1,
//                                ".png");
    testDetectEllipseExperiment(path_ellipse_centers_experiment_synthetic,
                                path_test_img_experiment_synthetic,
                                type_img_experiment_synthetic,
                                count_img_experiment_synthetic,
                                path_data_img_synthetic,
                                path_data_mask_synthetic,
                                type_img_experiment_synthetic,
                                type_img_experiment_synthetic);

    return 0;
}