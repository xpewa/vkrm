#include "detectEllipse.h"


//Ellipse DetectEllipse::__mnk(std::vector<Point> const& pts) {
//    Ellipse ellipse;
//    cv::Mat M(pts.size(), 5, CV_32F);
//    for (int i = 0; i < pts.size(); ++i) {
//        M.row(i).col(0) = pow(pts[i].x, 2);
//        M.row(i).col(1) = pts[i].x * pts[i].y;
//        M.row(i).col(2) = pow(pts[i].y, 2);
//        M.row(i).col(3) = pts[i].x;
//        M.row(i).col(4) = pts[i].y;
//    }
//    cv::Mat one = cv::Mat::ones(pts.size(), 1, CV_32F);
//    cv::Mat inv;
//    cv::invert(M.t() * M, inv);
//    cv::Mat res = inv * M.t() * one;
//    ellipse.A = res.at<float>(0, 0);
//    ellipse.B = res.at<float>(1, 0);
//    ellipse.C = res.at<float>(2, 0);
//    ellipse.D = res.at<float>(3, 0);
//    ellipse.E = res.at<float>(4, 0);
//    double F = -1;
//
//    double scale = F / ellipse.A;
//    ellipse.A *= scale;
//    ellipse.B *= scale;
//    ellipse.C *= scale;
//    ellipse.D *= scale;
//    ellipse.E *= scale;
//
//    double A = ellipse.A;
//    double B = ellipse.B;
//    double C = ellipse.C;
//    double D = ellipse.D;
//    double E = ellipse.E;
//
//    if (pow(B, 2) - 4*A*C) {
//        ellipse.x = (B*E - 2*C*D) / (4*A*C - pow(B, 2));
//        ellipse.y = (B*D - 2*A*E) / (4*A*C - pow(B, 2));
//        ellipse.angle = 0.5 * atan2(-B, C - A);
//        ellipse.R1 = -sqrt(2*(A * pow(E, 2) + C * pow(D, 2) -B*D*E + F*(pow(B, 2) - 4*A*C))*((A+C) - sqrt(pow((A - C), 2) + pow(B, 2)))) / (pow(B, 2) - 4*A*C);
//        ellipse.R2 = -sqrt(2*(A * pow(E, 2) + C * pow(D, 2) -B*D*E + F*(pow(B, 2) - 4*A*C))*((A+C) + sqrt(pow((A - C), 2) + pow(B, 2)))) / (pow(B, 2) - 4*A*C);
//    }
//    return ellipse;
//}

Ellipse DetectEllipse::__mnk(std::vector<Point> const& pts) {
    Ellipse ellipse;
    int numPoints = pts.size();

    Eigen::MatrixXd matA(numPoints, 5);
    Eigen::VectorXd vecb(numPoints);

    for (int i = 0; i < numPoints; ++i) {
        matA(i, 0) = pts[i].x * pts[i].x;  // x^2
        matA(i, 1) = pts[i].x * pts[i].y;  // xy
        matA(i, 2) = pts[i].y * pts[i].y;  // y^2
        matA(i, 3) = pts[i].x;              // x
        matA(i, 4) = pts[i].y;              // y
        vecb(i) = 1.0;                         // 1 (F = -1)
    }

    // Решение системы Ax = b методом наименьших квадратов
    Eigen::VectorXd x = (matA.transpose() * matA).ldlt().solve(matA.transpose() * vecb);

    ellipse.A = x(0);
    ellipse.B = x(1);
    ellipse.C = x(2);
    ellipse.D = x(3);
    ellipse.E = x(4);
    ellipse.F = -1.0;

    double A = ellipse.A;
    double B = ellipse.B;
    double C = ellipse.C;
    double D = ellipse.D;
    double E = ellipse.E;
    double F = ellipse.F;

    if (pow(B, 2) - 4*A*C) {
        ellipse.x = (B*E - 2*C*D) / (4*A*C - pow(B, 2));
        ellipse.y = (B*D - 2*A*E) / (4*A*C - pow(B, 2));
        ellipse.angle = 0.5 * atan2(-B, C - A) * 180 / CV_PI;
        ellipse.R1 = -sqrt(2*(A * pow(E, 2) + C * pow(D, 2) -B*D*E + F*(pow(B, 2) - 4*A*C))*((A+C) - sqrt(pow((A - C), 2) + pow(B, 2)))) / (pow(B, 2) - 4*A*C);
        ellipse.R2 = -sqrt(2*(A * pow(E, 2) + C * pow(D, 2) -B*D*E + F*(pow(B, 2) - 4*A*C))*((A+C) + sqrt(pow((A - C), 2) + pow(B, 2)))) / (pow(B, 2) - 4*A*C);
    }
    return ellipse;
}

double __findMode(const std::vector<double>& arr, double range_min = 0, double bin_size = 0.001) {
    std::map<int, int> frequencyMap;
    for (double num : arr) {
        int bin = static_cast<int>((num - range_min) / bin_size);
        frequencyMap[bin]++;
    }
    int mode_bin = frequencyMap.begin()->first;
    int maxFrequency = frequencyMap.begin()->second;
    for (const auto& pair : frequencyMap) {
        if (pair.second > maxFrequency) {
            maxFrequency = pair.second;
            mode_bin = pair.first;
        }
    }
    return range_min + (mode_bin + 0.5) * bin_size;
}

Ellipse DetectEllipse::detectEllipse(std::vector<Point> const& pts) { // rows x 2
    Ellipse ellipse;
    if (pts.empty()) return ellipse;

    auto start = std::chrono::high_resolution_clock::now();

//    int maxSize = 0;
    int maxSize = 1280;
    int countPointInRange = 100;
    for (int i = 0; i < pts.size(); ++i) {
        if (pts[i].x > maxSize) {
            maxSize = pts[i].x;
        }
        if (pts[i].y > maxSize) {
            maxSize = pts[i].y;
        }
    }

    int count_pts = pts.size();
    cv::Mat center = cv::Mat::zeros(maxSize + 10, maxSize + 10, CV_8U); // img size
    std::vector<double> angle_array, R1_array, R2_array;


    int maxIterations = 100;
    for (int i = 0; i < maxIterations; ++i) {
        std::vector<int> index(countPointInRange);
        for (int i = 0; i < countPointInRange; ++i) {
            index[i] = rand() % (count_pts);
        }
        std::vector<Point> points;
        for (int i = 0; i < countPointInRange; ++i) {
            points.push_back(pts[index[i]]);
        }

        Ellipse ellipse = __mnk(points);

        if (ellipse.y > 0 && ellipse.x > 0 && ellipse.y < maxSize && ellipse.x < maxSize) {
            angle_array.push_back(ellipse.angle);
            R1_array.push_back(ellipse.R1);
            R2_array.push_back(ellipse.R2);

            for (int y = ellipse.y - 1; y < ellipse.y + 2; ++y) {
                for (int x = ellipse.x - 1; x < ellipse.x + 2; ++x) {
                    if (x == ellipse.x && y == ellipse.y) {
                        center.row(y).col(x) += 2;
                    }
                    else {
                        center.row(y).col(x) += 1;
                    }
                }
            }
        }
    }

    double minVal;
    double maxVal;
    cv::Point minLoc;
    cv::Point maxLoc;
    minMaxLoc(center, &minVal, &maxVal, &minLoc, &maxLoc);

    ellipse.x = maxLoc.x;
    ellipse.y = maxLoc.y;
    ellipse.angle = __findMode(angle_array, -1, 1e-3);
    ellipse.R1 = __findMode(R1_array, 0, 1e-2);
    ellipse.R2 = __findMode(R2_array, 0, 1e-2);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
//    std::cout << "Время: " << duration.count() << " microseconds" << std::endl;

    return ellipse;
}
