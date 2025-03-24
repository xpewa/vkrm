#include "testCommon.h"

std::string PATH_CYLINDER = "../../cylinder.txt";
std::string PATH = "../../Experiment_synthetic_2/Image_";
std::string PATH_MASK = "../../Experiment_synthetic_2/mask/Image_";


std::vector<Ball> readGroundTruthFromFile(const std::string& filename) {
    std::vector<Ball> groundTruth;
    std::ifstream inputFile(filename);
    if (!inputFile.is_open()) {
        std::cerr << "Не удалось открыть файл: " << filename << std::endl;
        return groundTruth;
    }
    std::string line;
    while (std::getline(inputFile, line)) {
        std::stringstream ss(line);
        Ball point;
        if (ss >> point.x >> point.y >> point.z) {
            groundTruth.push_back(point);
        } else {
            std::cerr << "Ошибка при чтении строки: " << line << std::endl;
        }
    }
    inputFile.close();
    return groundTruth;
}


Ball calculateMAE(const std::vector<Ball>& predicted, const std::vector<Ball>& groundTruth) {
    if (predicted.size() != groundTruth.size() || predicted.empty()) {
        std::cerr << "Ошибка calculateMAE: Векторы должны быть одинакового размера и не должны быть пустыми." << std::endl;
        return {0.0, 0.0, 0.0};
    }
    size_t n = predicted.size();
    Ball MAE = {0.0, 0.0, 0.0};
    for (size_t i = 0; i < n; ++i) {
        MAE.x += std::abs(std::abs(predicted[i].x) - std::abs(groundTruth[i].x));
        MAE.y += std::abs(std::abs(predicted[i].y) - std::abs(groundTruth[i].y));
        MAE.z += std::abs(std::abs(predicted[i].z) - std::abs(groundTruth[i].z));
    }
    MAE.x /= n;
    MAE.y /= n;
    MAE.z /= n;
    return MAE;
}


Ball calculateMaxError(const std::vector<Ball>& predicted, const std::vector<Ball>& groundTruth) {
    if (predicted.size() != groundTruth.size() || predicted.empty()) {
        std::cerr << "Ошибка calculateMaxError: Векторы должны быть одинакового размера и не должны быть пустыми." << std::endl;
        return {0.0, 0.0, 0.0};
    }
    size_t n = predicted.size();
    Ball maxError = {0.0, 0.0, 0.0};
    for (size_t i = 0; i < n; ++i) {
        maxError.x = std::max(maxError.x, std::abs(std::abs(predicted[i].x) - std::abs(groundTruth[i].x)));
        maxError.y = std::max(maxError.y, std::abs(std::abs(predicted[i].y) - std::abs(groundTruth[i].y)));
        maxError.z = std::max(maxError.z, std::abs(std::abs(predicted[i].z) - std::abs(groundTruth[i].z)));
    }
    return maxError;
}


Ball calculateMinError(const std::vector<Ball>& predicted, const std::vector<Ball>& groundTruth) {
    if (predicted.size() != groundTruth.size() || predicted.empty()) {
        std::cerr << "Ошибка: Векторы должны быть одинакового размера и не должны быть пустыми." << std::endl;
        return {0.0, 0.0, 0.0};
    }
    size_t n = predicted.size();
    Ball minError = {std::numeric_limits<double>::max(),
                     std::numeric_limits<double>::max(),
                     std::numeric_limits<double>::max()};
    for (size_t i = 0; i < n; ++i) {
        minError.x = std::min(minError.x, std::abs(std::abs(predicted[i].x) - std::abs(groundTruth[i].x)));
        minError.y = std::min(minError.y, std::abs(std::abs(predicted[i].y) - std::abs(groundTruth[i].y)));
        minError.z = std::min(minError.z, std::abs(std::abs(predicted[i].z) - std::abs(groundTruth[i].z)));
    }
    return minError;
}


void testFindBallExperiment(std::string path_ellipse_centers, std::string path_ball_positions, std::string path_test_img, std::string type_img, int count_img) {
    std::vector<Ellipse> real_centers = readCentersFromFile(path_ellipse_centers);
    std::vector<Ellipse> predict_centers;
    std::vector<Ball> ball_array;

    std::vector<Ball> real_ball = readGroundTruthFromFile(path_ball_positions);

    ColorFilter colorFilter;
    Cylinder cylinder;
    cylinder.load(PATH_CYLINDER);
    if (cylinder.R) {
        colorFilter = ColorFilter(cylinder);
    }
    else {
        cylinder = colorFilter.train(PATH, PATH_MASK, ".png", ".png", count_img);
        cylinder.save(PATH_CYLINDER);
    }
    FindBall findBall(colorFilter);

    std::vector<long long> times;
    for (int i = 1; i < count_img + 1; ++i) {
        cv::Mat img_test = cv::imread(path_test_img + std::to_string(i) + type_img);

        auto start = std::chrono::high_resolution_clock::now();

        Ellipse ellipse = findBall.getEllipseParameters(img_test);

        Ball ball = findBall.findBall(img_test);

        auto end = std::chrono::high_resolution_clock::now();

        predict_centers.push_back(ellipse);
        ball_array.push_back(ball);

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        times.push_back(duration.count());
    }

    double averageTime = accumulate(times.begin(), times.end(), 0.0) / times.size();
    std::cout << "Среднее время обработки изображения: " << averageTime << " microseconds" << std::endl;

    Ball MAE = calculateMAE(ball_array, real_ball);
    Ball maxError = calculateMaxError(ball_array, real_ball);
    Ball minError = calculateMinError(ball_array, real_ball);

    std::cout << "MAE: (" << MAE.x << ", " << MAE.y << ", " << MAE.z << ")" << std::endl;
    std::cout << "Max Error: (" << maxError.x << ", " << maxError.y << ", " << maxError.z << ")" << std::endl;
    std::cout << "Min Error: (" << minError.x << ", " << minError.y << ", " << minError.z << ")" << std::endl;

//    for (int i = 0; i < predict_centers.size(); ++i) {
//        std::cout << "predict_centers " << i << ": " << predict_centers[i].x << " " << predict_centers[i].y << std::endl;
//    }
//    for (int i = 0; i < ball_array.size(); ++i) {
//        std::cout << "ball_array " << i << ": " << ball_array[i].x << " " << ball_array[i].y << " " << ball_array[i].z << std::endl;
//    }
}


int main() {
    std::string path_ellipse_centers_experiment_synthetic = "../../Experiment_synthetic_2/ball_center.txt";
    std::string path_ball_positions_experiment_synthetic = "../../Experiment_synthetic_2/ball_position.txt";
    std::string path_test_img_experiment_synthetic = "../../Experiment_synthetic_2/Image_";
    std::string path_mask_img_experiment_synthetic = "../../Experiment_synthetic_2/mask/Image_";
    std::string type_img_experiment_synthetic = ".png";
    int count_img_experiment_synthetic = 660;


    testFindBallExperiment(path_ellipse_centers_experiment_synthetic, path_ball_positions_experiment_synthetic, path_test_img_experiment_synthetic, type_img_experiment_synthetic, count_img_experiment_synthetic);
    return 0;
}
