#include "testCommon.h"

std::string PATH_CYLINDER = "../../cylinder.txt";
std::string PATH = "../../Experiment_synthetic/Image_";
std::string PATH_MASK = "../../Experiment_synthetic/mask/Image_";


void testFindBallExperiment(std::string path_ellipse_centers, std::string path_test_img, std::string type_img, int count_img) {
    std::vector<Ellipse> real_centers = readCentersFromFile(path_ellipse_centers);
    std::vector<Ellipse> predict_centers;
    std::vector<Ball> ball_array;

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

    for (int i = 0; i < predict_centers.size(); ++i) {
        std::cout << "predict_centers " << i << ": " << predict_centers[i].x << " " << predict_centers[i].y << std::endl;
    }
    for (int i = 0; i < ball_array.size(); ++i) {
        std::cout << "ball_array " << i << ": " << ball_array[i].x << " " << ball_array[i].y << " " << ball_array[i].z << std::endl;
    }
}


int main() {
    std::string path_ellipse_centers_experiment_synthetic = "../../Experiment_synthetic/ellipse_centers.txt";
    std::string path_test_img_experiment_synthetic = "../../Experiment_synthetic/Image_";
    std::string path_mask_img_experiment_synthetic = "../../Experiment_synthetic/mask/Image_";
    std::string type_img_experiment_synthetic = ".png";
    int count_img_experiment_synthetic = 1;


    testFindBallExperiment(path_ellipse_centers_experiment_synthetic, path_test_img_experiment_synthetic, type_img_experiment_synthetic, count_img_experiment_synthetic);
    return 0;
}
