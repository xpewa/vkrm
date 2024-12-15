#include "detectEllipse.h"


Ellipse DetectEllipse::__mnk(std::vector<Point> const& pts) {
    Ellipse ellipse;
    cv::Mat M(pts.size(), 5, CV_32F);
    for (int i = 0; i < pts.size(); ++i) {
        M.row(i).col(0) = pow(pts[i].x, 2);
        M.row(i).col(1) = pts[i].x * pts[i].y;
        M.row(i).col(2) = pow(pts[i].y, 2);
        M.row(i).col(3) = pts[i].x;
        M.row(i).col(4) = pts[i].y;
    }
    cv::Mat one = cv::Mat::ones(pts.size(), 1, CV_32F);
    cv::Mat inv;
    cv::invert(M.t() * M, inv);
    cv::Mat res = inv * M.t() * one;
    ellipse.A = res.at<float>(0, 0);
    ellipse.B = res.at<float>(1, 0);
    ellipse.C = res.at<float>(2, 0);
    ellipse.D = res.at<float>(3, 0);
    ellipse.E = res.at<float>(4, 0);

    if (4*ellipse.A*ellipse.C - pow(ellipse.B, 2))
        ellipse.x = (ellipse.B*ellipse.E - 2*ellipse.C*ellipse.D) / (4*ellipse.A*ellipse.C - pow(ellipse.B, 2));
    if (4*ellipse.A*ellipse.C - pow(ellipse.B, 2))
        ellipse.y = (ellipse.B*ellipse.D - 2*ellipse.A*ellipse.E) / (4*ellipse.A*ellipse.C - pow(ellipse.B, 2));
    return ellipse;
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

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
//    std::cout << "Время: " << duration.count() << " microseconds" << std::endl;

    return ellipse;
}
