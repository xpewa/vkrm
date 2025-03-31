#include "findBall.h"

// ----------------------------------- GET ELLIPSE ----------------------------------------------

Ellipse FindBall::getEllipseParameters(cv::Mat const & img) {
    cv::Mat resize_img = img.clone();
    int img_width = img.cols;
    int img_height = img.rows;
    resize(img, resize_img, cv::Size(img_width / scale, img_height / scale), cv::INTER_LINEAR);

//    cv::imshow("resize_img", resize_img);
//    cv::waitKey(0);
    cv::Mat img_res = colorFilter.recognize(resize_img);
//    cv::imshow("colorFilter", img_res);
//    cv::waitKey(0);
    EdgeDetection edgeDetection(MIN_SIZE_OBJECT, MAX_SIZE_OBJECT);
    std::vector<Point> imagePoints = edgeDetection.find_points(img_res);
    edgePoints = imagePoints;
//    cv::Mat emptyImg = cv::Mat::zeros(cv::Size(img_res.cols, img_res.rows),CV_8UC1);
//    emptyImg = edgeDetection.draw_points(emptyImg, imagePoints);
//    cv::imshow("edgeDetection", emptyImg);
//    cv::waitKey(0);
    DetectEllipse detectEllipse;
    Ellipse ellipse_res = detectEllipse.detectEllipse(imagePoints);
    ellipse = ellipse_res;
//    cv::Point centerCircle1(ellipse.x, ellipse.y);
//    cv::Scalar colorCircle1(0, 0, 255);
//    cv::circle(resize_img, centerCircle1, 3, colorCircle1, cv::FILLED);
//    cv::imshow("img res", resize_img);
//    cv::waitKey(0);

    ellipse.x *= scale;
    ellipse.y *= scale;
    ellipse.R1 *= scale;
    ellipse.R2 *= scale;

    return ellipse;
}

// ----------------------------------- USEFUL ----------------------------------------------

double deg2rad(double deg) {
    return deg * M_PI / 180.0;
}

// ----------------------------------- CAMERA ----------------------------------------------


Vec3 Camera::cameraToWorld(const Vec3 & Pc) {
    // Pc = R * Pw + T,
    // Pw = R^T * (Pc - T)
    Vec3 diff;
    diff.x = Pc.x - T.x;
    diff.y = Pc.y - T.y;
    diff.z = Pc.z - T.z;
    Vec3 Pw;
    // Умножаем на транспонированную матрицу R (столбцы становятся строками)
    Pw.x = R[0][0] * diff.x + R[1][0] * diff.y + R[2][0] * diff.z;
    Pw.y = R[0][1] * diff.x + R[1][1] * diff.y + R[2][1] * diff.z;
    Pw.z = R[0][2] * diff.x + R[1][2] * diff.y + R[2][2] * diff.z;
    return Pw;
}


void Camera::computeCameraExtrinsics() {
    // Переводим углы в радианы
    double rx = deg2rad(rotx);
    double ry = deg2rad(roty);
    double rz = deg2rad(rotz);

    // Вычисляем матрицы поворота по каждой оси
    double Rx[3][3] = {
            {1,         0,          0},
            {0, cos(rx), -sin(rx)},
            {0, sin(rx),  cos(rx)}
    };
    double Ry[3][3] = {
            { cos(ry), 0, sin(ry)},
            {      0,  1,      0},
            {-sin(ry), 0, cos(ry)}
    };
    double Rz[3][3] = {
            {cos(rz), -sin(rz), 0},
            {sin(rz),  cos(rz), 0},
            {     0,       0,   1}
    };

    // R = Rz * Ry * Rx.
    // R_temp = Ry * Rx:
    double R_temp[3][3] = {0};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            R_temp[i][j] = 0;
            for (int k = 0; k < 3; k++) {
                R_temp[i][j] += Ry[i][k] * Rx[k][j];
            }
        }
    }
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            R[i][j] = 0;
            for (int k = 0; k < 3; k++) {
                R[i][j] += Rz[i][k] * R_temp[k][j];
            }
        }
    }

    T = {transx, transy, transz};
}


// ----------------------------------- FIND BALL ----------------------------------------------

Ball FindBall::findBall(cv::Mat const & img) {
//    cv::imshow("initial img", img);
//    cv::waitKey(0);
    Ellipse _ellipse = getEllipseParameters(img);
    Camera camera;
    Ball ball;
    auto start = std::chrono::high_resolution_clock::now();
    ellipse.R1 *= 0.91;
    ellipse.R2 *= 0.91;
    ball = estimate3dCoords_1();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
//    std::cout << "Среднее время обработки изображения estimate3dCoords_1: " << duration.count() << " microseconds" << std::endl;
//    std::cout << "distance in mm: " << 50 * 120 / (ellipse.R1 * 0.025 ) << std::endl;
    return ball;
}

// ----------------------------------- USEFUL FOR FIND BALL 1 and 3 ----------------------------------------------

Vec3 FindBall::pixelToRay(double u, double v) {
    double x = (u - camera.cx) / camera.fx; // (u - cx) / fx
    double y = (v - camera.cy) / camera.fy; // (v - cy) / fy
    double z = 1.0;

    Vec3 rayDirection = {x, y, z};
    return rayDirection.normalize();
}


std::vector<Point> FindBall::get2DPointsFromEllipse(int numPoints) {
    std::vector<Point> ellipsePoints;
    for (int i = 0; i < numPoints; ++i) {
        double angle = 2 * M_PI * i / numPoints;
        float x = ellipse.R1 * cos(angle);
        float y = ellipse.R2 * sin(angle);
        cv::Mat rotationMatrix = cv::getRotationMatrix2D(cv::Point2f(0,0), ellipse.angle, 1.0);
        cv::Mat point2D = (cv::Mat_<double>(3, 1) << x, y, 1);
        cv::Mat rotatedPoint = rotationMatrix * point2D;
        ellipsePoints.push_back(Point(rotatedPoint.at<double>(0) + ellipse.x, rotatedPoint.at<double>(1) + ellipse.y));
    }
//    cv::Mat new_img = cv::Mat::zeros(cv::Size(1280, 1024),CV_8UC1);
//    for (int i = 0; i < ellipsePoints.size(); ++i) {
//        Point p = ellipsePoints[i];
//        cv::Point centerCircle(p.x, p.y);
//        cv::Scalar colorCircle(255);
//        cv::circle(new_img, centerCircle, 1, colorCircle, cv::FILLED);
//    }
//    cv::imshow("ellipsePoints", new_img);
//    cv::waitKey(0);
    return ellipsePoints;
}


Vec3 findNormalLeastSquares(const std::vector<Vec3>& points) {
    if (points.size() < 3) {
        std::cerr << "findNormalLeastSquares: Недостаточно точек для определения плоскости" << std::endl;
        return {0.0, 0.0, 0.0};
    }
    Eigen::Vector3d centroid(0.0, 0.0, 0.0);
    for (const auto& p : points) {
        centroid(0) += p.x;
        centroid(1) += p.y;
        centroid(2) += p.z;
    }
    centroid /= points.size();
    Eigen::MatrixXd A(points.size(), 3);
    for (size_t i = 0; i < points.size(); ++i) {
        A(i, 0) = points[i].x - centroid(0);
        A(i, 1) = points[i].y - centroid(1);
        A(i, 2) = points[i].z - centroid(2);
    }
    Eigen::Matrix3d C = A.transpose() * A;
    Eigen::EigenSolver<Eigen::Matrix3d> eigenSolver(C);
    Eigen::Vector3d eigenvalues = eigenSolver.eigenvalues().real();
    Eigen::Matrix3d eigenvectors = eigenSolver.eigenvectors().real();
    int minEigenvalueIndex = 0;
    for (int i = 1; i < 3; ++i) {
        if (eigenvalues(i) < eigenvalues(minEigenvalueIndex)) {
            minEigenvalueIndex = i;
        }
    }
    Eigen::Vector3d normalEigen = eigenvectors.col(minEigenvalueIndex);
    Eigen::Vector3d normalEigenNormalized = normalEigen.normalized();
    Vec3 normal;
    normal.x = normalEigenNormalized(0);
    normal.y = normalEigenNormalized(1);
    normal.z = normalEigenNormalized(2);
    return normal;
}


Vec3 findNormalFromThreePoints(const Vec3& p1, const Vec3& p2, const Vec3& p3) {
    Vec3 v1 = {p2.x - p1.x, p2.y - p1.y, p2.z - p1.z};
    Vec3 v2 = {p3.x - p1.x, p3.y - p1.y, p3.z - p1.z};
    Vec3 norm = v1.cross(v2).normalize();
    return norm;
}


Vec3 FindBall::getNormalForPlane(const std::vector<Vec3>& points) {
    int count_random = 100000;

    Vec3 averageNormal = {0.0, 0.0, 0.0};
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, points.size() - 1);

    for (int i = 0; i < count_random; ++i) {
        int index1 = distrib(gen);
        int index2 = distrib(gen);
        int index3 = distrib(gen);
        // Убеждаемся, что все индексы разные
        while (index2 == index1) {
            index2 = distrib(gen);
        }
        while (index3 == index1 || index3 == index2) {
            index3 = distrib(gen);
        }
        Vec3 p1 = points[index1];
        Vec3 p2 = points[index2];
        Vec3 p3 = points[index3];
        Vec3 normal = findNormalFromThreePoints(p1, p2, p3);
        // Проверяем, направлена ли новая нормаль в ту же сторону, что и средняя нормаль
        if (i > 0 && normal.dot(averageNormal) < 0) {
            normal.x = -normal.x;
            normal.y = -normal.y;
            normal.z = -normal.z;
        }
        averageNormal.x += normal.x;
        averageNormal.y += normal.y;
        averageNormal.z += normal.z;
    }
    averageNormal.x /= count_random;
    averageNormal.y /= count_random;
    averageNormal.z /= count_random;

    averageNormal = averageNormal.normalize();

    if (averageNormal.z < 0) {
        averageNormal.x = -averageNormal.x;
        averageNormal.y = -averageNormal.y;
        averageNormal.z = -averageNormal.z;
    }

    return averageNormal;
}


double FindBall::computeConeAngle(const std::vector<Vec3>& edgeVectors, const Vec3& planeNormal) {
    std::vector<double> angles;
    double sumAngles = 0.0;
    for (const auto& v : edgeVectors) {
        double cosTheta = v.dot(planeNormal);
//        std::cout << "cosAngle: " << cosTheta << std::endl;
        cosTheta = std::max(-1.0, std::min(1.0, cosTheta));
        double theta = std::acos(cosTheta);
        angles.push_back(theta);
        sumAngles += theta;
    }
    double maxAngle = 0.0;
    for (double angle : angles) {
        maxAngle = std::max(maxAngle, angle);
    }
    return 2.0 * std::min(maxAngle, M_PI - maxAngle); // * 180 / M_PI
//    return 2.0 * std::min((sumAngles / edgeVectors.size()), M_PI - (sumAngles / edgeVectors.size()));
}


Vec3 FindBall::calculateBallCenter13(const Vec3& planeNormal, double coneApexAngle) {
    double halfAngle = coneApexAngle / 2.0;
    double distance = BALL_RADIUS / std::sin(halfAngle);
//    std::cout << "distance: " << distance << std::endl;
    Vec3 result;
    Vec3 normPlaneNormal = planeNormal.normalize();
    result.x = distance * normPlaneNormal.x;
    result.y = distance * normPlaneNormal.y;
    result.z = distance * normPlaneNormal.z;
    return result;
}


std::vector<Point> FindBall::getContourPoints(const cv::Mat& image, int n) {
    std::vector<Point> points;
    // координаты белых пикселей
    std::vector<Point> whitePixels;
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            if (image.at<uchar>(y, x) > 0) {
                whitePixels.push_back({x, y});
            }
        }
    }
    if (whitePixels.empty()) {
        std::cerr << "Нет белых пикселей" << std::endl;
        return points;
    }
    // центр масс белых пикселей
    Point center(0, 0);
    for (const auto& p : whitePixels) {
        center.x += p.x;
        center.y += p.y;
    }
    center.x /= whitePixels.size();
    center.y /= whitePixels.size();
    // среднее расстояние от центра
    double outerRadius = 0;
    for (const auto& p : whitePixels) {
        outerRadius += std::sqrt(std::pow(p.x - center.x, 2) + std::pow(p.y - center.y, 2));
    }
    outerRadius /= whitePixels.size();
    // граничные пиксели
    std::vector<Point> boundaryPixels;
    double radiusTolerance = 0.2 * outerRadius; // Погрешность для определения радиуса
    for (Point& p : whitePixels) {
        double distanceToCenter = std::sqrt(std::pow(p.x - center.x, 2) + std::pow(p.y - center.y, 2));
        bool isBoundary = false;
        if (p.x > 0 && image.at<uchar>(p.y, p.x - 1) == 0) isBoundary = true;
        if (p.x < image.cols - 1 && image.at<uchar>(p.y, p.x + 1) == 0) isBoundary = true;
        if (p.y > 0 && image.at<uchar>(p.y - 1, p.x) == 0) isBoundary = true;
        if (p.y < image.rows - 1 && image.at<uchar>(p.y + 1, p.x) == 0) isBoundary = true;

        if (isBoundary && std::abs(distanceToCenter - outerRadius) < radiusTolerance) {
//        if (isBoundary && distanceToCenter > outerRadius) {
            boundaryPixels.push_back(p);
        }
    }
    // сортировка граничных пикселей по углу относительно центра
    std::sort(boundaryPixels.begin(), boundaryPixels.end(), [&](const Point& a, const Point& b) {
        double angleA = std::atan2(a.y - center.y, a.x - center.x);
        double angleB = std::atan2(b.y - center.y, b.x - center.x);
        return angleA < angleB;
    });
    // выбор n точек, равномерно распределенных по отсортированному списку
    if (boundaryPixels.size() < n) {
        std::cerr << "Недостаточно точек на контуре" << std::endl;
        for (const Point& p : boundaryPixels) {
            points.push_back(p);
        }
    } else {
        double step = static_cast<double>(boundaryPixels.size()) / n;
        for (int i = 0; i < n; ++i) {
            int index = static_cast<int>(std::round(i * step));
            points.push_back(boundaryPixels[index]);
        }
    }
    return points;
}


// ----------------------------------- FIND BALL 1 ----------------------------------------------


Ball FindBall::estimate3dCoords_1() {
    std::vector<Point> edgeVectors2D = get2DPointsFromEllipse(3);
    std::vector<Vec3> edgeVectors;
    for (const auto& pixel : edgeVectors2D) {
        Vec3 ray = pixelToRay(pixel.x, pixel.y);
        edgeVectors.push_back(ray);
    }
//    std::cout << "3d вектора:" << std::endl;
//    for (const auto& vec : edgeVectors) {
//        std::cout << "[" << vec.x << ", " << vec.y << ", " << vec.z << "]" << std::endl;
//    }

    Vec3 planeNormal = findNormalFromThreePoints(edgeVectors[0], edgeVectors[1], edgeVectors[2]);
//    std::cout << "Нормаль: ("
//              << planeNormal.x << ", " << planeNormal.y << ", " << planeNormal.z << ")\n";

    double coneApexAngle = computeConeAngle(edgeVectors, planeNormal);
//    std::cout << "Угол при вершине конуса: " << coneApexAngle << std::endl;

    Vec3 Pc = calculateBallCenter13(planeNormal, coneApexAngle);

    Ball ball = {Pc.x, Pc.y, Pc.z};
    return ball;
}

// ----------------------------------- FIND BALL 3 ----------------------------------------------


Ball FindBall::estimate3dCoords_3() {
    cv::Mat img_with_edgePoints = cv::Mat::zeros(cv::Size(camera.width, camera.height),CV_8UC1);
    for (int i = 0; i < edgePoints.size(); ++i) {
        edgePoints[i].x *= scale;
        edgePoints[i].y *= scale;
    }
    EdgeDetection edgeDetection(MIN_SIZE_OBJECT, MAX_SIZE_OBJECT);
    img_with_edgePoints = edgeDetection.draw_points(img_with_edgePoints, edgePoints);
//    cv::imshow("edgeDetection", img_with_edgePoints);
//    cv::waitKey(0);

    std::vector<Point> points = getContourPoints(img_with_edgePoints, 100);
//    img_with_edgePoints = cv::Mat::zeros(cv::Size(camera.width, camera.height),CV_8UC1);
//    img_with_edgePoints = edgeDetection.draw_points(img_with_edgePoints, points);
//    cv::imshow("edgeDetection points", img_with_edgePoints);
//    cv::waitKey(0);

    std::vector<Vec3> edgeVectors;
    for (const auto& pixel : points) {
        Vec3 ray = pixelToRay(pixel.x, pixel.y);
        edgeVectors.push_back(ray);
    }
//    std::cout << "3d вектора:" << std::endl;
//    for (const auto& vec : edgeVectors) {
//        std::cout << "[" << vec.x << ", " << vec.y << ", " << vec.z << "]" << std::endl;
//    }
//    Vec3 planeNormal = getNormalForPlane(edgeVectors);
    Vec3 planeNormal = findNormalLeastSquares(edgeVectors);
//    std::cout << "Нормаль: ("
//              << planeNormal.x << ", " << planeNormal.y << ", " << planeNormal.z << ")\n";
    double coneApexAngle = computeConeAngle(edgeVectors, planeNormal);
//    std::cout << "Угол при вершине конуса: " << coneApexAngle << std::endl;
    Vec3 Pc = calculateBallCenter13(planeNormal, coneApexAngle);
    Ball ball = {Pc.x, Pc.y, Pc.z};
    return ball;
}


// ----------------------------------- USEFUL FOR FIND BALL 2 and 4 ----------------------------------------------

Vec3 FindBall::calculateBallCenter2() {
    double mean_R = (ellipse.R2 + ellipse.R1) / 2;
    cv::Vec3d b_c_2d = {double(ellipse.x), double(ellipse.y), 1};
    cv::Vec3d e_c_1_2d = {double(ellipse.x), double(ellipse.y - mean_R), 1};
    cv::Vec3d e_c_2_2d = {double(ellipse.x), double(ellipse.y + mean_R), 1};

    double m[3][3] = {{camera.fx, 0.0, camera.cx}, {0.0, camera.fy, camera.cy}, {0.0, 0.0, 1.0}};
    cv::Mat K = cv::Mat(3, 3, CV_64F, m);

    cv::Mat b_c = K.inv() * b_c_2d;
    cv::Mat e_c_1 = K.inv() * e_c_1_2d;
    cv::Mat e_c_2 = K.inv() * e_c_2_2d;

    cv::Mat e_c = e_c_2 - e_c_1;

    cv::Mat b = ((BALL_RADIUS * 2) * b_c) / e_c.at<double>(0, 1);
    Vec3 ballCenter = {b.at<double>(0, 0), b.at<double>(0, 1), b.at<double>(0, 2)};
    return ballCenter;
}


Vec3 FindBall::calculateBallCenter4(const std::vector<Point>& points) {
    Point center(0, 0);
    for (const auto& p : points) {
        center.x += p.x;
        center.y += p.y;
    }
    center.x /= points.size();
    center.y /= points.size();
    double mean_R = 0;
    for (const auto& p : points) {
        mean_R += std::sqrt(std::pow(p.x - center.x, 2) + std::pow(p.y - center.y, 2));
    }
    mean_R /= points.size();

    cv::Vec3d b_c_2d = {double(center.x), double(center.y), 1};
    cv::Vec3d e_c_1_2d = {double(center.x), double(center.y - mean_R), 1};
    cv::Vec3d e_c_2_2d = {double(center.x), double(center.y + mean_R), 1};

    double m[3][3] = {{camera.fx, 0.0, camera.cx}, {0.0, camera.fy, camera.cy}, {0.0, 0.0, 1.0}};
    cv::Mat K = cv::Mat(3, 3, CV_64F, m);

    cv::Mat b_c = K.inv() * b_c_2d;
    cv::Mat e_c_1 = K.inv() * e_c_1_2d;
    cv::Mat e_c_2 = K.inv() * e_c_2_2d;

    cv::Mat e_c = e_c_2 - e_c_1;

    cv::Mat b = ((BALL_RADIUS * 2) * b_c) / e_c.at<double>(0, 1);
    Vec3 ballCenter = {b.at<double>(0, 0), b.at<double>(0, 1), b.at<double>(0, 2)};
    return ballCenter;
}


// ----------------------------------- FIND BALL 2 ----------------------------------------------

Ball FindBall::estimate3dCoords_2() {
    Vec3 Pc = calculateBallCenter2();
    Ball ball = {Pc.x, Pc.y, Pc.z};
    return ball;
}

// ----------------------------------- FIND BALL 4 ----------------------------------------------


Ball FindBall::estimate3dCoords_4() {
    cv::Mat img_with_edgePoints = cv::Mat::zeros(cv::Size(camera.width, camera.height),CV_8UC1);
    for (int i = 0; i < edgePoints.size(); ++i) {
        edgePoints[i].x *= scale;
        edgePoints[i].y *= scale;
    }
    EdgeDetection edgeDetection(MIN_SIZE_OBJECT, MAX_SIZE_OBJECT);
    img_with_edgePoints = edgeDetection.draw_points(img_with_edgePoints, edgePoints);
//    cv::imshow("edgeDetection", img_with_edgePoints);
//    cv::waitKey(0);

    std::vector<Point> points = getContourPoints(img_with_edgePoints, 3);
//    img_with_edgePoints = cv::Mat::zeros(cv::Size(camera.width, camera.height),CV_8UC1);
//    img_with_edgePoints = edgeDetection.draw_points(img_with_edgePoints, points);
//    cv::imshow("edgeDetection points", img_with_edgePoints);
//    cv::waitKey(0);

    Vec3 Pc = calculateBallCenter4(points);
    Ball ball = {Pc.x, Pc.y, Pc.z};
    return ball;
}
