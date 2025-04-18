#include "detectEllipse.h"


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

    int maxSize = 1280;
    int countPointInRange = 100;

    int count_pts = pts.size();
    cv::Mat center = cv::Mat::zeros(maxSize + 10, maxSize + 10, CV_8U);
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

    std::sort(angle_array.begin(), angle_array.end());
    std::sort(R1_array.begin(), R1_array.end());
    std::sort(R2_array.begin(), R2_array.end());

    ellipse.x = maxLoc.x;
    ellipse.y = maxLoc.y;

    ellipse.angle = angle_array[int(angle_array.size() / 2)];
    ellipse.R1 = R1_array[int(R1_array.size() / 2)];
    ellipse.R2 = R2_array[int(R2_array.size() / 2)];

    return ellipse;
}



Ellipse DetectEllipse::detectEllipse_2(std::vector<Point> const& pts) {
    double centerX = 0;
    double centerY = 0;

    for (const auto& point : pts) {
        centerX += point.x;
        centerY += point.y;
    }

    centerX /= pts.size();
    centerY /= pts.size();

    std::vector<Distance> distances;
    for (int index = 0; index < pts.size(); ++index) {
        const auto& point = pts[index];
        Distance distance;

        double deltaX = point.x - centerX;
        double deltaY = point.y - centerY;

        distance.distance = std::pow(deltaX, 2) + std::pow(deltaY, 2);
        distance.angle = std::atan2(deltaY, deltaX);
        distance.index = index;
        distances.push_back(distance);
    }

    Ellipse ellipse;
    ellipse.x = centerX;
    ellipse.y = centerY;
    return ellipse;
}


Ellipse DetectEllipse::detectEllipse_3(std::vector<Point> const& pts) {
    double sumX = 0.0;
    double sumY = 0.0;
    for (const auto& point : pts) {
        sumX += point.x;
        sumY += point.y;
    }
    Point center;
    center.x = sumX / pts.size();
    center.y = sumY / pts.size();

    std::vector<Point> centeredPoints;
    for (const auto& point : pts) {
        centeredPoints.push_back({point.x - center.x, point.y - center.y});
    }

    Eigen::Matrix2d P;
    P.setZero();
    for (const auto& point : centeredPoints) {
        Eigen::Vector2d p(point.x, point.y);
        P += p * p.transpose();
    }

    Eigen::EigenSolver<Eigen::Matrix2d> solver(P);
    Eigen::Vector2d eigenvalues = solver.eigenvalues().real();
    Eigen::Matrix2d eigenvectors = solver.eigenvectors().real();

    int majorAxisIndex = (eigenvalues(0) > eigenvalues(1)) ? 0 : 1;
    int minorAxisIndex = 1 - majorAxisIndex;

    Eigen::Vector2d majorAxis = eigenvectors.col(majorAxisIndex);
    Eigen::Vector2d minorAxis = eigenvectors.col(minorAxisIndex);

    double angle = std::atan2(majorAxis(1), majorAxis(0));  // radians

    double majorAxisLength = std::sqrt(eigenvalues(majorAxisIndex) / pts.size() * 2);
    double minorAxisLength = std::sqrt(eigenvalues(minorAxisIndex) / pts.size() * 2);

    Ellipse ellipse;
    ellipse.x = center.x;
    ellipse.y = center.y;
    ellipse.angle = angle;
    ellipse.R1 = majorAxisLength;
    ellipse.R2 = minorAxisLength;

    return ellipse;
}
