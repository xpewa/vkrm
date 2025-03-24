#include "findBall.h"


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
    Ellipse res_ellipse = detectEllipse.detectEllipse(imagePoints);
    ellipse = res_ellipse;
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


//cv::Vec3d findPlaneNormal(const std::vector<cv::Vec3d>& vectors) {
//    cv::Vec3d sum_of_vectors(0, 0, 0);
//    for (const auto& vec : vectors) {
//        double magnitude = sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
//        if (magnitude == 0) continue;
//        sum_of_vectors += vec / magnitude;
//    }
//
//    cv::Vec3d average_vector = sum_of_vectors / static_cast<double>(vectors.size());
//
//    double average_magnitude = sqrt(average_vector[0] * average_vector[0] +
//                                    average_vector[1] * average_vector[1] +
//                                    average_vector[2] * average_vector[2]);
//
//    if (average_magnitude == 0)
//        return cv::Vec3d(0,0,0);
//
//    return average_vector / average_magnitude;
//}
//
//double findConeApexAngle(const std::vector<cv::Vec3d>& points, const cv::Vec3d& normal) {
//    double sumAngles = 0.0;
//    double normalMagnitude = sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);
//
//    for (const auto& point : points) {
//        double dotProduct = point[0] * normal[0] + point[1] * normal[1] + point[2] * normal[2];
//        double pointMagnitude = sqrt(point[0]*point[0] + point[1]*point[1] + point[2]*point[2]);
//
//        double cosAngle = dotProduct / (normalMagnitude * pointMagnitude);
//
//        if (abs(cosAngle) > 1.0) {
//            cosAngle = (cosAngle > 0) ? 1.0 : -1.0;
//        }
//
//        sumAngles += acos(cosAngle);
//    }
//    return 2.0 * sumAngles / points.size(); // в радианах
//}
//
//// Функция для вычисления 3D координат мяча
//cv::Vec3d findBallCoordinates(const cv::Vec3d& normal, double coneAngle, double ballRadius) {
//    double sinHalfAngle = sin(coneAngle / 2.0);
//    double normalMagnitude = sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);
//    double distance = ballRadius / sinHalfAngle;
//
//    std::cout << "distance: " << distance << std::endl;
//
//    return (distance / normalMagnitude) * normal;
//}
//
//// Функция для создания набора 3D точек из эллипса на изображении
//std::vector<cv::Vec3d> get3DPointsFromEllipse(const Camera& camera, const Ellipse& ellipse, int numPoints) {
//    std::vector<cv::Vec3d> points3D;
//
//    std::vector<cv::Point2f> ellipsePoints;
//    for (int i = 0; i < numPoints; ++i) {
//        double angle = 2 * M_PI * i / numPoints;
//        float x = ellipse.R1 * cos(angle);
//        float y = ellipse.R2 * sin(angle);
//
//        // относительно нуля
//        cv::Mat rotationMatrix = cv::getRotationMatrix2D(cv::Point2f(0,0), ellipse.angle, 1.0);
//
//        // относительно нуля
//        cv::Mat point2D = (cv::Mat_<double>(3, 1) << x, y, 1);
//
//        cv::Mat rotatedPoint = rotationMatrix * point2D;
//
//        // Получаем точку после поворота, и смещаем ее.
//        ellipsePoints.push_back(cv::Point2f(rotatedPoint.at<double>(0) + ellipse.x, rotatedPoint.at<double>(1) + ellipse.y));
//    }
//    cv::Mat new_img = cv::Mat::zeros(cv::Size(1280, 1024),CV_8UC1);
//    for (int i = 0; i < ellipsePoints.size(); ++i) {
//        cv::Point2f p = ellipsePoints[i];
//        cv::Point centerCircle(p.x, p.y);
//        cv::Scalar colorCircle(255);
//        cv::circle(new_img, centerCircle, 1, colorCircle, cv::FILLED);
//    }
//    cv::imshow("ellipsePoints", new_img);
//    cv::waitKey(0);
//
//    // 2. Обратная проекция каждой точки в 3D
//    for (const auto& point2D : ellipsePoints) {
//        std::vector<cv::Point2f> distorted_points = {point2D};
//        std::vector<cv::Point2f> undistorted_points;
//        cv::undistortPoints(distorted_points, undistorted_points, camera.get_camera_matrix(), camera.get_distortion_coeff());
//        cv::Point2f undistorted_point = undistorted_points[0];
////        cv::Point2f undistorted_point = point2D;
//
//        cv::Mat point3DHomogeneous = (cv::Mat_<double>(3,1) << undistorted_point.x, undistorted_point.y, 1);
//
//        points3D.push_back(cv::Vec3d(point3DHomogeneous.at<double>(0),
//                                     point3DHomogeneous.at<double>(1),
//                                     point3DHomogeneous.at<double>(2)));
//    }
//    return points3D;
//}
//
//cv::Vec3d transformVectorCameraToWorld(const Camera& camera, const cv::Vec3d& vector) {
//    cv::Mat vector_homogeneous = (cv::Mat_<double>(4, 1) << vector[0], vector[1], vector[2], 1);
//
//    std::cout << camera.get_world_to_camera_matrix() << std::endl;
//
//    cv::Mat camera_to_world_matrix = camera.get_world_to_camera_matrix().inv(cv::DECOMP_SVD);
//    cv::Mat camera_to_world =  (cv::Mat_<double>(4, 4) <<
//            0.6947, -0.3040, 0.6519, 2.0000,
//            0.7193,  0.2936, -0.6296, -2.0000,
//            0.0000,  0.9063,  0.4226,  1.0000,
//            0.0000,  0.0000,  0.0000,  1.0000);
//
//    cv::Mat world_vector_homogeneous = camera_to_world * vector_homogeneous;
//    cv::Vec3d world_vector(world_vector_homogeneous.at<double>(0),
//                       world_vector_homogeneous.at<double>(1),
//                       world_vector_homogeneous.at<double>(2));
//    return world_vector;
//}
//
//
//Ball FindBall::estimate3dCoords(Ellipse ellipse, Camera camera) {
//
//    std::vector<cv::Vec3d> points = get3DPointsFromEllipse(camera, ellipse, 10);
//
//    cv::Vec3d normal = findPlaneNormal(points);
//    std::cout << "normal: " << normal << std::endl;
//
//    double coneAngle = findConeApexAngle(points, normal);
//    std::cout << "Cone Apex Angle: " << coneAngle * 180 / M_PI << " degrees" << std::endl;
//
//    cv::Vec3d ballCoordinates = findBallCoordinates(normal, coneAngle, BALL_RADIUS);
//    ballCoordinates = transformVectorCameraToWorld(camera, ballCoordinates);
//
//    Ball ball;
//    ball.x = ballCoordinates[0];
//    ball.y = ballCoordinates[1];
//    ball.z = ballCoordinates[2];
//    return ball;
//}


Ball FindBall::findBall(cv::Mat const & img) {
//    cv::imshow("initial img", img);
//    cv::waitKey(0);

    Ellipse ellipse = getEllipseParameters(img);

    Camera camera;

    Ball ball;
    auto start = std::chrono::high_resolution_clock::now();
    ball = estimate3dCoords_4(ellipse, camera);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
//    std::cout << "Среднее время обработки изображения: " << duration.count() << " microseconds" << std::endl;

//    std::cout << "HI, Alina, you distance in mm: " << 50 * 120 / (ellipse.R1 * 0.025 ) << std::endl;

    return ball;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////


//std::vector<Point2D> generateEllipsePoints(const Ellipse& ellipse, int numPoints) {
//    std::vector<Point2D> points;
////    if (n <= 0) return points;
////
////    double angleStep = 2 * M_PI / n;
////    for (int i = 0; i < n; ++i) {
////        double theta = i * angleStep;
////        double x = ellipse.x + ellipse.R1 * cos(theta) * cos(ellipse.angle) - ellipse.R2 * sin(theta) * sin(ellipse.angle);
////        double y = ellipse.y + ellipse.R1 * cos(theta) * sin(ellipse.angle) + ellipse.R2 * sin(theta) * cos(ellipse.angle);
////        points.push_back({x, y});
////    }
////    return points;
//
//    double theta = ellipse.angle * CV_PI / 180.0;
//    double a = ellipse.R1;
//    double b = ellipse.R2;
//
//    for (int i = 0; i < numPoints; ++i) {
//        double phi = 2 * CV_PI * i / numPoints;
//        double x = a * cos(phi);
//        double y = b * sin(phi);
//        double x_rot = x * cos(theta) - y * sin(theta);
//        double y_rot = x * sin(theta) + y * cos(theta);
//        points.push_back({ellipse.x + x_rot, ellipse.y + y_rot});
//    }
//    return points;
//}
//
//// Функция для нормализации вектора
//Eigen::Vector3d normalize(const Eigen::Vector3d& v) {
//    return v / v.norm();
//}
//
//// Функция для вычисления луча и его единичного вектора для точки на изображении
//Eigen::Vector3d computeRay(const Point2D& u, const Camera& camera) {
//    Eigen::Vector3d u_homogenous(u.x, u.y, 1.0);
//    Eigen::Vector3d ray_camera_space = camera.K.inverse() * u_homogenous;
//    Eigen::Vector3d ray_world_space = camera.rotation * ray_camera_space;
//    Eigen::Vector3d unit_vector = normalize(ray_world_space);
//
////    std::cout << "u_homogenous: " << u_homogenous << std::endl;
//    return unit_vector;
//}
//
//std::vector<Eigen::Vector3d> computeRayForEllipsePoints(const std::vector<Point2D>& points, const Camera& camera) {
//    std::vector<Eigen::Vector3d> arrayOfRays;
//    for (int i = 0; i < points.size(); ++i) {
//        arrayOfRays.push_back(computeRay(points[i], camera));
//    }
//    return arrayOfRays;
//}
//
//Eigen::Vector3d findBestFittingPlaneNormal(const std::vector<Eigen::Vector3d>& vectors) {
//    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
//    for (const auto& v : vectors) {
//        A += v * v.transpose();
//    }
//
//    Eigen::EigenSolver<Eigen::Matrix3d> eigenSolver(A);
//    Eigen::Vector3d eigenvalues = eigenSolver.eigenvalues().real();
//    Eigen::Matrix3d eigenvectors = eigenSolver.eigenvectors().real();
//
//    // Индекс наименьшего собственного значения
//    int minEigenvalueIndex = 0;
//    for (int i = 1; i < 3; ++i) {
//        if (eigenvalues(i) < eigenvalues(minEigenvalueIndex)) {
//            minEigenvalueIndex = i;
//        }
//    }
//
//    // Нормаль к наилучшей плоскости - это собственный вектор, соответствующий наименьшему собственному значению.
//    Eigen::Vector3d normal = eigenvectors.col(minEigenvalueIndex).normalized();
//    return normal;
//}
//
////Eigen::Vector3d findBestFittingPlaneNormalSVD(const std::vector<Eigen::Vector3d>& vectors) {
////    Eigen::MatrixXf A(vectors.size(), 3);
////    for (size_t i = 0; i < vectors.size(); ++i) {
////        A.row(i) = vectors[i].transpose();
////    }
////
////    Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
////    Eigen::VectorXf singularValues = svd.singularValues();
////    Eigen::MatrixXf V = svd.matrixV();
////
////    // Находим индекс наименьшего сингулярного значения
////    int minSingularValueIndex = 0;
////    for (int i = 1; i < 3; ++i) {
////        if (singularValues(i) < singularValues(minSingularValueIndex)) {
////            minSingularValueIndex = i;
////        }
////    }
////
////    // Нормаль к наилучшей плоскости - это последний столбец матрицы V.
////    Eigen::Vector3d normal = V.col(minSingularValueIndex);
////    return normal.normalized();
////}
//
//double calculateAverageConeAngle(const std::vector<Eigen::Vector3d>& vectors, const Eigen::Vector3d& normal) {
//    double sum_of_angles = 0.0;
//    for (const auto& v : vectors) {
//        // Убедимся, что векторы нормализованы
//        Eigen::Vector3d normalized_v = v.normalized();
//
////        std::cout << "normal.dot(normalized_v): " << normal.dot(normalized_v) << std::endl;
//
//        // Вычисляем угол между вектором и нормалью
//        double angle = std::acos(normal.dot(normalized_v));
//
//        // Добавляем угол к сумме
//        sum_of_angles += angle;
//    }
//
//    // Вычисляем средний угол
//    double average_angle = sum_of_angles / vectors.size();
//
//    return 2 * average_angle;
//}
//
//Ball FindBall::estimate3dCoords(Ellipse ellipse, Camera camera) {
//    std::vector<Point2D> ellipsePoints = generateEllipsePoints(ellipse, 10);
////    cv::Mat new_img = cv::Mat::zeros(cv::Size(1280, 1024),CV_8UC1);
////    for (int i = 0; i < ellipsePoints.size(); ++i) {
////        Point2D p = ellipsePoints[i];
////        cv::Point centerCircle(p.x, p.y);
////        cv::Scalar colorCircle(255);
////        cv::circle(new_img, centerCircle, 1, colorCircle, cv::FILLED);
////    }
////    cv::imshow("ellipsePoints", new_img);
////    cv::waitKey(0);
//
//    std::vector<Eigen::Vector3d> rays = computeRayForEllipsePoints(ellipsePoints, camera);
////    for (int i = 0; i < rays.size(); ++i) {
////        std::cout << "ray " << i << " : " << rays[i] << std::endl;
////    }
//
//    Eigen::Vector3d normal = findBestFittingPlaneNormal(rays);
////    std::cout << "normal : " << normal << std::endl;
//
//    double angle = calculateAverageConeAngle(rays, normal);
//    std::cout << "angle : " << angle << std::endl;
//
//    Ball ball;
//    return ball;
//}

////////////////////////////////////////////////////////////////////////////////////

//// Структура для параметров конуса
//struct Cone {
//    cv::Vec3d axis;     // Ось конуса (единичный вектор)
//    double angle;   // Угол раствора в радианах
//};
//
//Cone estimateConeFromEllipse(const Ellipse& ellipse, const cv::Mat& cameraMatrix) {
//    // 1. Преобразование эллипса в нормализованные координаты
//    cv::Point2d center = (cv::Point2f(ellipse.x, ellipse.y) - cv::Point2f(cameraMatrix.at<double>(0,2),
//                                               cameraMatrix.at<double>(1,2)));
//    center.x /= cameraMatrix.at<double>(0,0);
//    center.y /= cameraMatrix.at<double>(1,1);
//
//    // 2. Построение матрицы конуса
//    double a = ellipse.R1 / cameraMatrix.at<double>(0,0);
//    double b = ellipse.R2 / cameraMatrix.at<double>(1,1);
//    double theta = ellipse.angle * CV_PI/180.0;
//
//    double cos_t = cos(theta);
//    double sin_t = sin(theta);
//
//    cv::Mat Q = (cv::Mat_<double>(3,3) <<
//                               pow(cos(theta),2)/pow(a,2) + pow(sin(theta),2)/pow(b,2),
//            sin(2*theta)*(1/pow(a,2) - 1/pow(b,2))/2,
//            -center.x,
//
//            sin(2*theta)*(1/pow(a,2) - 1/pow(b,2))/2,
//            pow(sin(theta),2)/pow(a,2) + pow(cos(theta),2)/pow(b,2),
//            -center.y,
//
//            -center.x,
//            -center.y,
//            pow(center.x,2) + pow(center.y,2) - 1
//    );
//
//    // 3. Собственное разложение для нахождения оси конуса
//    cv::Mat eigenvalues, eigenvectors;
//    eigen(Q, eigenvalues, eigenvectors);
//
//    // Выбор правильного собственного вектора
//    int idx = eigenvalues.at<double>(0) < eigenvalues.at<double>(1) ? 0 : 1;
//    idx = eigenvalues.at<double>(idx) < eigenvalues.at<double>(2) ? idx : 2;
//
//    cv::Vec3d axis = eigenvectors.row(idx);
//    if(axis[2] > 0) axis = -axis; // Инвертируем направление для системы OpenCV
//
//    // Расчёт угла
//    double cos_angle = 1.0 / sqrt(axis.dot(axis));
//    cos_angle = cv::max(-1.0, cv::min(1.0, cos_angle));
//    double angle = acos(cos_angle);
//
//    // Нормализация оси
//    axis /= norm(axis);
//
//    return {axis, angle};
//}
//
//Ball FindBall::estimate3dCoords(Ellipse ellipse, Camera camera) {
////    std::cout << "Эллипс center: " << ellipse.x << " " << ellipse.y << std::endl;
////    std::cout << "Эллипс R1 R2: " << ellipse.R1 << " " << ellipse.R2 << std::endl;
////    std::cout << "Эллипс angle: " << ellipse.angle << std::endl;
//
//    // Получаем параметры конуса
//    Cone cone = estimateConeFromEllipse(ellipse, camera.get_camera_matrix());
//
////    // Рассчитываем расстояние до центра сферы
////    double d = BALL_RADIUS / sin(cone.angle);
//// Расчёт расстояния (с учётом системы координат Blender)
//    double d = BALL_RADIUS / tan(cone.angle);
//
//    // Рассчитываем 3D координаты
//    cv::Vec3d sphere_center = d * cone.axis;
//
//    std::cout << "Sphere center in camera coordinates: " << sphere_center << std::endl;
//    std::cout << "Distance: " << d << " meters" << std::endl;
//
//
//    // Преобразование в систему координат Blender
//    cv::Mat R = (cv::Mat_<double>(3,3) <<
//                               0, 0, 1,
//            -1, 0, 0,
//            0, -1, 0
//    );
//
//    cv::Vec3d sphere_blender = R * sphere_center;
//
//    std::cout << "OpenCV Camera System:\n";
//    std::cout << "Normalized axis: " << cone.axis << std::endl;
//    std::cout << "Cone angle: " << cone.angle*180/CV_PI << " deg\n";
//    std::cout << "Distance: " << d << " m\n";
//    std::cout << "Sphere center (OpenCV): " << sphere_center << std::endl << std::endl;
//
////    std::cout << "Blender System:\n";
////    std::cout << "Sphere position: " << sphere_blender << std::endl;
//
//    Ball ball;
//    return ball;
//}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

//struct Vec3 {
//    double x, y, z;
//};
//
//struct CameraIntrinsics {
//    double f;   // пиксели
//    double cx;  //
//    double cy;  //
//};
//
//struct CameraExtrinsics {
//    double R[3][3];
//    Vec3 T;
//};


//// Функция для отображения набора 2D точек из эллипса на изображении
//void show2DPointsFromEllipse(const Ellipse& ellipse, int numPoints) {
//    std::vector<cv::Point2f> ellipsePoints;
//    for (int i = 0; i < numPoints; ++i) {
//        double angle = 2 * M_PI * i / numPoints;
//        float x = ellipse.R1 * cos(angle);
//        float y = ellipse.R2 * sin(angle);
//
//        cv::Mat rotationMatrix = cv::getRotationMatrix2D(cv::Point2f(0,0), ellipse.angle, 1.0);
//
//        cv::Mat point2D = (cv::Mat_<double>(3, 1) << x, y, 1);
//
//        cv::Mat rotatedPoint = rotationMatrix * point2D;
//
//        ellipsePoints.push_back(cv::Point2f(rotatedPoint.at<double>(0) + ellipse.x, rotatedPoint.at<double>(1) + ellipse.y));
//    }
//    cv::Mat new_img = cv::Mat::zeros(cv::Size(1280, 1024),CV_8UC1);
//    for (int i = 0; i < ellipsePoints.size(); ++i) {
//        cv::Point2f p = ellipsePoints[i];
//        cv::Point centerCircle(p.x, p.y);
//        cv::Scalar colorCircle(255);
//        cv::circle(new_img, centerCircle, 1, colorCircle, cv::FILLED);
//    }
//    cv::imshow("ellipsePoints", new_img);
//    cv::waitKey(0);
//}


//Vec3 computeSphereCenterInCameraCoordinates(double u0, double v0,
//                                            double a, double b,
//                                            double ballRadius,
//                                            const CameraIntrinsics &intr) {
//    // средний радиус
//    double r_img = (a + b) / 2.0;
//
//    // расстояние от камеры до центра сферы:
//    // d = r * sqrt(f^2 + r_img^2) / r_img
//    double d = ballRadius * std::sqrt(intr.f * intr.f + r_img * r_img) / r_img;
//
//    // Нормализуем координаты центра изображения (относительно главной точки)
//    double x_norm = (u0 - intr.cx) / intr.f;
//    double y_norm = (v0 - intr.cy) / intr.f;
//
//    // Так как проекция устроена как X = x_norm * Z, Y = y_norm * Z, выбираем Z = d.
//    Vec3 Pc;
//    Pc.x = x_norm * d;
//    Pc.y = y_norm * d;
//    Pc.z = d;
//
//    return Pc;
//}
//Vec3 cameraToWorld(const Vec3 &Pc, const CameraExtrinsics &extr) {
//    // Если камера задана соотношением: Pc = R * Pw + T,
//    // то Pw = R^T * (Pc - T)
//    Vec3 diff;
//    diff.x = Pc.x - extr.T.x;
//    diff.y = Pc.y - extr.T.y;
//    diff.z = Pc.z - extr.T.z;
//
//    Vec3 Pw;
//    // Умножаем на транспонированную матрицу R (столбцы становятся строками)
//    Pw.x = extr.R[0][0] * diff.x + extr.R[1][0] * diff.y + extr.R[2][0] * diff.z;
//    Pw.y = extr.R[0][1] * diff.x + extr.R[1][1] * diff.y + extr.R[2][1] * diff.z;
//    Pw.z = extr.R[0][2] * diff.x + extr.R[1][2] * diff.y + extr.R[2][2] * diff.z;
//
//    return Pw;
//}
//
//// Функция для перевода градусов в радианы
//double deg2rad(double deg) {
//    return deg * M_PI / 180.0;
//}


//void computeCameraExtrinsics(double camX, double camY, double camZ,
//                             double rotX_deg, double rotY_deg, double rotZ_deg,
//                             CameraExtrinsics &extr)
//{
//    // Переводим углы в радианы
//    double rx = deg2rad(rotX_deg);
//    double ry = deg2rad(rotY_deg);
//    double rz = deg2rad(rotZ_deg);
//
//    // Вычисляем матрицы поворота по каждой оси
//    // Ротация вокруг оси X
//    double Rx[3][3] = {
//            {1,         0,          0},
//            {0, cos(rx), -sin(rx)},
//            {0, sin(rx),  cos(rx)}
//    };
//
//    // Ротация вокруг оси Y
//    double Ry[3][3] = {
//            { cos(ry), 0, sin(ry)},
//            {      0,  1,      0},
//            {-sin(ry), 0, cos(ry)}
//    };
//
//    // Ротация вокруг оси Z
//    double Rz[3][3] = {
//            {cos(rz), -sin(rz), 0},
//            {sin(rz),  cos(rz), 0},
//            {     0,       0,   1}
//    };
//
//    // Предполагаем порядок R = Rz * Ry * Rx.
//    // Сначала вычисляем R_temp = Ry * Rx:
//    double R_temp[3][3] = {0};
//    for (int i = 0; i < 3; i++) {
//        for (int j = 0; j < 3; j++) {
//            R_temp[i][j] = 0;
//            for (int k = 0; k < 3; k++) {
//                R_temp[i][j] += Ry[i][k] * Rx[k][j];
//            }
//        }
//    }
//    // Затем вычисляем итоговую матрицу R = Rz * R_temp:
//    for (int i = 0; i < 3; i++) {
//        for (int j = 0; j < 3; j++) {
//            extr.R[i][j] = 0;
//            for (int k = 0; k < 3; k++) {
//                extr.R[i][j] += Rz[i][k] * R_temp[k][j];
//            }
//        }
//    }
//
//    // Задаём центр камеры в мировой системе:
//    Vec3 C = { camX, camY, camZ };
//
//    extr.T = C;
//
////    // Вычисляем T = -R * C
////    extr.T.x = -(extr.R[0][0]*C.x + extr.R[0][1]*C.y + extr.R[0][2]*C.z);
////    extr.T.y = -(extr.R[1][0]*C.x + extr.R[1][1]*C.y + extr.R[1][2]*C.z);
////    extr.T.z = -(extr.R[2][0]*C.x + extr.R[2][1]*C.y + extr.R[2][2]*C.z);
//}
//
//Ball FindBall::estimate3dCoords(Ellipse ellipse, Camera camera) {
//    show2DPointsFromEllipse(ellipse, 100);
//
//    double u0 = ellipse.x;
//    double v0 = ellipse.y;
//    double a  = ellipse.R1;
//    double b  = ellipse.R2;
//    double ballRadius = BALL_RADIUS;
//
//    CameraIntrinsics intr;
//    intr.f  = camera.fx;    // пиксели
//    intr.cx = camera.cx;
//    intr.cy = camera.cy;
//
//    // 3D координаты центра мяча в системе камеры
//    Vec3 Pc = computeSphereCenterInCameraCoordinates(u0, v0, a, b, ballRadius, intr);
//    std::cout << "Центр мяча в координатах камеры: ("
//              << Pc.x << ", " << Pc.y << ", " << Pc.z << ")\n";
//
////    CameraExtrinsics extr;
////    extr.R[0][0] = 0.6947; extr.R[0][1] = -0.3040; extr.R[0][2] = 0.6519;
////    extr.R[1][0] = 0.7193; extr.R[1][1] = 0.2936; extr.R[1][2] = -0.6296;
////    extr.R[2][0] = 0; extr.R[2][1] = 0.9063; extr.R[2][2] = 0.45399;
////    extr.T.x = 1; extr.T.y = -2; extr.T.z = 1;
//
//    CameraExtrinsics extr;
//
//    computeCameraExtrinsics(0, 0, 0, 120, 0, -30, extr);
//
////    std::cout << "Матрица поворота R:" << std::endl;
////    for (int i = 0; i < 3; i++) {
////        std::cout << extr.R[i][0] << "\t"
////                  << extr.R[i][1] << "\t"
////                  << extr.R[i][2] << std::endl;
////    }
////    std::cout << "Вектор T: ("
////              << extr.T.x << ", "
////              << extr.T.y << ", "
////              << extr.T.z << ")" << std::endl;
//
//    Vec3 Pw = cameraToWorld(Pc, extr);
//    std::cout << "Центр мяча в мировой системе координат: ("
//              << Pw.x << ", " << Pw.y << ", " << Pw.z << ")\n";
//
//    Ball ball;
//    return ball;
//}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Vec3 cameraToWorld(const Vec3 &Pc, const CameraExtrinsics &extr) {
    // Pc = R * Pw + T,
    // Pw = R^T * (Pc - T)
    Vec3 diff;
    diff.x = Pc.x - extr.T.x;
    diff.y = Pc.y - extr.T.y;
    diff.z = Pc.z - extr.T.z;

    Vec3 Pw;
    // Умножаем на транспонированную матрицу R (столбцы становятся строками)
    Pw.x = extr.R[0][0] * diff.x + extr.R[1][0] * diff.y + extr.R[2][0] * diff.z;
    Pw.y = extr.R[0][1] * diff.x + extr.R[1][1] * diff.y + extr.R[2][1] * diff.z;
    Pw.z = extr.R[0][2] * diff.x + extr.R[1][2] * diff.y + extr.R[2][2] * diff.z;

    return Pw;
}

// Функция для отображения набора 2D точек из эллипса на изображении
std::vector<cv::Point2f> get2DPointsFromEllipse(const Ellipse& ellipse, int numPoints) {
    std::vector<cv::Point2f> ellipsePoints;
    for (int i = 0; i < numPoints; ++i) {
        double angle = 2 * M_PI * i / numPoints;
        float x = ellipse.R1 * cos(angle);
        float y = ellipse.R2 * sin(angle);

        cv::Mat rotationMatrix = cv::getRotationMatrix2D(cv::Point2f(0,0), ellipse.angle, 1.0);

        cv::Mat point2D = (cv::Mat_<double>(3, 1) << x, y, 1);

        cv::Mat rotatedPoint = rotationMatrix * point2D;

        ellipsePoints.push_back(cv::Point2f(rotatedPoint.at<double>(0) + ellipse.x, rotatedPoint.at<double>(1) + ellipse.y));
    }
    cv::Mat new_img = cv::Mat::zeros(cv::Size(1280, 1024),CV_8UC1);
    for (int i = 0; i < ellipsePoints.size(); ++i) {
        cv::Point2f p = ellipsePoints[i];
        cv::Point centerCircle(p.x, p.y);
        cv::Scalar colorCircle(255);
        cv::circle(new_img, centerCircle, 1, colorCircle, cv::FILLED);
    }
//    cv::imshow("ellipsePoints", new_img);
//    cv::waitKey(0);
    return ellipsePoints;
}


// Функция для перевода градусов в радианы
double deg2rad(double deg) {
    return deg * M_PI / 180.0;
}


void computeCameraExtrinsics(double camX, double camY, double camZ,
                             double rotX_deg, double rotY_deg, double rotZ_deg,
                             CameraExtrinsics &extr)
{
    // Переводим углы в радианы
    double rx = deg2rad(rotX_deg);
    double ry = deg2rad(rotY_deg);
    double rz = deg2rad(rotZ_deg);

    // Вычисляем матрицы поворота по каждой оси
    // Ротация вокруг оси X
    double Rx[3][3] = {
            {1,         0,          0},
            {0, cos(rx), -sin(rx)},
            {0, sin(rx),  cos(rx)}
    };

    // Ротация вокруг оси Y
    double Ry[3][3] = {
            { cos(ry), 0, sin(ry)},
            {      0,  1,      0},
            {-sin(ry), 0, cos(ry)}
    };

    // Ротация вокруг оси Z
    double Rz[3][3] = {
            {cos(rz), -sin(rz), 0},
            {sin(rz),  cos(rz), 0},
            {     0,       0,   1}
    };

    // Предполагаем порядок R = Rz * Ry * Rx.
    // Сначала вычисляем R_temp = Ry * Rx:
    double R_temp[3][3] = {0};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            R_temp[i][j] = 0;
            for (int k = 0; k < 3; k++) {
                R_temp[i][j] += Ry[i][k] * Rx[k][j];
            }
        }
    }
    // Затем вычисляем итоговую матрицу R = Rz * R_temp:
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            extr.R[i][j] = 0;
            for (int k = 0; k < 3; k++) {
                extr.R[i][j] += Rz[i][k] * R_temp[k][j];
            }
        }
    }

    // Задаём центр камеры в мировой системе:
    Vec3 C = { camX, camY, camZ };

    extr.T = C;

//    // Вычисляем T = -R * C
//    extr.T.x = -(extr.R[0][0]*C.x + extr.R[0][1]*C.y + extr.R[0][2]*C.z);
//    extr.T.y = -(extr.R[1][0]*C.x + extr.R[1][1]*C.y + extr.R[1][2]*C.z);
//    extr.T.z = -(extr.R[2][0]*C.x + extr.R[2][1]*C.y + extr.R[2][2]*C.z);
}


// Функция для аппроксимации плоскости по набору 3D-векторов (единичных векторов, направленных от камеры к границе сферы)
//Vec3 fitPlane(const std::vector<Vec3>& edgeVectors) {
//    // Создаем матрицу A, где каждая строка - это единичный вектор
//    Eigen::MatrixXd A(edgeVectors.size(), 3);
//    for (size_t i = 0; i < edgeVectors.size(); ++i) {
//        A(i, 0) = edgeVectors[i].x; // !!!!!!!!!!!!!!!!!!!!!!!!!
//        A(i, 1) = edgeVectors[i].y;
//        A(i, 2) = edgeVectors[i].z;
//    }
//
//    // SVD
//    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
//    Eigen::Vector3d col_2 = svd.matrixV().col(2); // Нормаль - это последний столбец V
//    col_2 = col_2.normalized(); // Нормализация
//    Vec3 normal;
//    normal.x = col_2(2); // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//    normal.y = col_2(1);
//    normal.z = col_2(0);
//
//    return normal; // Нормализуем нормаль
//}

// Функция для вычисления скалярного произведения векторов
double dotProduct(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Функция для вычисления длины вектора
double vectorLength(const Vec3& v) {
    return std::sqrt(dotProduct(v, v));
}

// Функция для нормализации вектора
Vec3 normalize(const Vec3& v) {
    double length = vectorLength(v);
    Vec3 result;
    if (length > 1e-9) { // Avoid division by zero
        result.x = v.x / length;
        result.y = v.y / length;
        result.z = v.z / length;
    } else {
        result.x = 0.0;
        result.y = 0.0;
        result.z = 1.0; // Или другой вектор по умолчанию
    }
    return result;
}

// Функция для вычисления векторного произведения
Vec3 crossProduct(const Vec3& a, const Vec3& b) {
    Vec3 result;
    result.x = a.y * b.z - a.z * b.y;
    result.y = a.z * b.x - a.x * b.z;
    result.z = a.x * b.y - a.y * b.x;
    return result;
}

Vec3 fitPlane(const std::vector<Vec3>& vectors) {
    cv::Vec3d sum_of_vectors(0, 0, 0);
    for (const auto& vec : vectors) {
        double magnitude = sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
        if (magnitude == 0) continue;
        cv::Vec3d vec3D = {vec.x, vec.y, vec.z};
        sum_of_vectors += vec3D / magnitude;
    }

    cv::Vec3d average_vector = sum_of_vectors / static_cast<double>(vectors.size());

    double average_magnitude = sqrt(average_vector[0] * average_vector[0] +
                                    average_vector[1] * average_vector[1] +
                                    average_vector[2] * average_vector[2]);

    if (average_magnitude == 0)
        return {0,0,0};

    Vec3 res = {average_vector[0] / average_magnitude, average_vector[1] / average_magnitude, average_vector[2] / average_magnitude};

    return res;
}


Vec3 normalizeVec3(const Vec3& vec) {
    Vec3 output;
    double mod = 0.0;
    mod += vec.x * vec.x;
    mod += vec.y * vec.y;
    mod += vec.z * vec.z;
    double mag = std::sqrt(mod);
    if (mag == 0) {
        throw std::logic_error("The input normalizeVec3 vector is a zero vector");
    }
    output.x = vec.x / mod;
    output.y = vec.y / mod;
    output.z = vec.z / mod;
    return output;
}


// Функция для вычисления угла при вершине конуса
//double computeConeApexAngle(const std::vector<Vec3>& edgeVectors, const Vec3& planeNormal) {
//    double sumOfAngles = 0.0;
//    for (const auto& v : edgeVectors) {
//        double cosAngle = dotProduct(planeNormal, normalizeVec3(v));
//        std::cout << "cosAngle: " << cosAngle << std::endl;
//        cosAngle = std::min(1.0, std::max(-1.0, cosAngle));
//        sumOfAngles += acos(cosAngle);
//    }
//    return 2.0 * (sumOfAngles / edgeVectors.size());
//}

// Функция для вычисления угла при вершине конуса
double computeConeApexAngle(const std::vector<Vec3>& edgeVectors, const Vec3& planeNormal) {
    if (edgeVectors.empty()) {
        return 0.0;
    }
    std::vector<double> angles;
    double sumAngles = 0.0;
    for (const auto& v : edgeVectors) {
        double cosTheta = dotProduct(v, planeNormal);
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
//    return 2.0 * std::min(maxAngle, M_PI - maxAngle); // * 180 / M_PI
    return 2.0 * std::min((sumAngles / edgeVectors.size()), M_PI - (sumAngles / edgeVectors.size()));
}


// Функция для вычисления координат центра сферы
Vec3 computeSphereCenter(double sphereRadius, const Vec3& planeNormal, double coneApexAngle) {
//    std::cout << "sphereRadius: " << sphereRadius << std::endl;
//    std::cout << "coneApexAngle: " << coneApexAngle << std::endl;
    double halfAngle = coneApexAngle / 2.0;
    double distance = sphereRadius / std::sin(halfAngle);
//   double distance = 50 * 120 / (ellipse.R1 * 0.025 ) / 1000;
//    double distance = ellipse.R1 / std::sin(halfAngle);
//    std::cout << "distance: " << distance << std::endl;
    Vec3 result;
    Vec3 normPlaneNormal = normalizeVec3(planeNormal);
//    normPlaneNormal.x = -normPlaneNormal.x;
//    normPlaneNormal.y = -normPlaneNormal.y;
//    normPlaneNormal.z = -normPlaneNormal.z;
//    std::cout << "normPlaneNormal: ("
//              << normPlaneNormal.x << ", " << normPlaneNormal.y << ", " << normPlaneNormal.z << ")\n";
    result.x = distance * normPlaneNormal.x;
    result.y = distance * normPlaneNormal.y;
    result.z = distance * normPlaneNormal.z;
    return result;
}

//Vec3 calculateBallCenter(Ellipse ellipse, const std::vector<Vec3>& edgeVectors, const Vec3& coneAxis, const Vec3& cameraPosition, double ballRadius) {
//    if (edgeVectors.size() < 2) return {0,0,0};
//
//    double min_x = 100000000;
//    int min_x_i = -10;
//    double max_x = -10000000;
//    int max_x_i = -10;
//    double min_y = 100000000;
//    int min_y_i = -10;
//    double max_y = -10000000;
//    int max_y_i = -10;
//    for (int i = 0; i < edgeVectors.size(); ++i) {
//        if (edgeVectors[i].x < min_x) {
//            min_x = edgeVectors[i].x;
//            min_x_i = i;
//        }
//        if (edgeVectors[i].x > max_x) {
//            max_x = edgeVectors[i].x;
//            max_x_i = i;
//        }
//        if (edgeVectors[i].y < min_y) {
//            min_y = edgeVectors[i].y;
//            min_y_i = i;
//        }
//        if (edgeVectors[i].y > max_y) {
//            max_y = edgeVectors[i].y;
//            max_y_i = i;
//        }
//    }
//    Vec3 e1_x = edgeVectors[min_x_i]; // K^-1 * [bx + d/2, by, 1]^T
//    Vec3 e2_x = edgeVectors[max_x_i]; // K^-1 * [bx - d/2, by, 1]^T
//    Vec3 e1_y = edgeVectors[min_y_i]; // K^-1 * [bx + d/2, by, 1]^T
//    Vec3 e2_y = edgeVectors[max_y_i]; // K^-1 * [bx - d/2, by, 1]^T
//
//    std::cout << "e_y_1: ("
//              << e1_y.x << ", " << e1_y.y << ", " << e1_y.z << ")\n";
//    std::cout << "e_y_2: ("
//              << e2_y.x << ", " << e2_y.y << ", " << e2_y.z << ")\n";
//
////    Vec3 e_x = {abs(e1_x.x - e2_x.x), abs(e1_x.y - e2_x.y), abs(e1_x.z - e2_x.z)};
////    Vec3 e_y = {abs(e1_y.x - e2_y.x), abs(e1_y.y - e2_y.y), abs(e1_y.z - e2_y.z)};
//    Vec3 e_x = {abs(e1_x.x - e2_x.x), 0, 0};
//    Vec3 e_y = {0, abs(e1_y.y - e2_y.y), 0};
//
//    Vec3 e = {abs(e_x.x - e_y.x), abs(e_x.y - e_y.y), abs(e_x.z - e_y.z)};
////    std::cout << "e: ("
////              << e.x << ", " << e.y << ", " << e.z << ")\n";
//    std::cout << "vectorLength: (" << vectorLength(e_y) << ")\n";
////    std::cout << "ellipse.R1: (" << ellipse.R1 << ")\n";
////    std::cout << "ellipse.R2: (" << ellipse.R2 << ")\n";
//
//
//    double lambda = ballRadius * 2.0 / vectorLength(e_y);
////    double lambda = ballRadius * 2.0 / ellipse.R2;
//
//    Vec3 ballCenter;
//    ballCenter.x = cameraPosition.x + coneAxis.x * lambda;
//    ballCenter.y = cameraPosition.y + coneAxis.y * lambda;
//    ballCenter.z = cameraPosition.z + coneAxis.z * lambda;
//
//    return ballCenter;
//}



// Функция для преобразования 2D-точки в 3D-луч
Vec3 pixelToRay(double u, double v, const Eigen::Matrix3d& K) {
    // Обратное преобразование из координат пикселей в нормализованные координаты изображения:
    double x = (u - K(0, 2)) / K(0, 0); // (u - cx) / fx
    double y = (v - K(1, 2)) / K(1, 1); // (v - cy) / fy
    double z = 1.0;

    Vec3 rayDirection = {x, y, z};
    return normalizeVec3(rayDirection);
//    return rayDirection;
}


Vec3 calculateBallCenter(Ellipse ellipse, double ballRadius, const cv::Mat& K) {
    double mean_R = (ellipse.R2 + ellipse.R1) / 2;
    cv::Vec3d b_c_2d = {double(ellipse.x), double(ellipse.y), 1};
    cv::Vec3d e_c_1_2d = {double(ellipse.x), double(ellipse.y - mean_R), 1};
    cv::Vec3d e_c_2_2d = {double(ellipse.x), double(ellipse.y + mean_R), 1};

    cv::Mat b_c = K.inv() * b_c_2d;
    cv::Mat e_c_1 = K.inv() * e_c_1_2d;
    cv::Mat e_c_2 = K.inv() * e_c_2_2d;

    cv::Mat e_c = e_c_2 - e_c_1;

    cv::Mat b = ((ballRadius * 2) * b_c) / e_c.at<double>(0, 1);
    Vec3 ballCenter = {b.at<double>(0, 0), b.at<double>(0, 1), b.at<double>(0, 2)};
    return ballCenter;
}


Vec3 calculateBallCenter(std::vector<Point> const& points, double ballRadius, const cv::Mat& K) {
    Point center(0, 0);
    for (const auto& p : points) {
        center.x += p.x;
        center.y += p.y;
    }
    if (!points.empty()) {
        center.x /= points.size();
        center.y /= points.size();
    } else {
        std::cerr << "Нет белых пикселей. Невозможно найти контур." << std::endl;
    }

    double mean_R = 0;
    for (const auto& p : points) {
        mean_R += std::sqrt(std::pow(p.x - center.x, 2) + std::pow(p.y - center.y, 2));
    }
    if (!points.empty()) {
        mean_R /= points.size();
    } else {
        std::cerr << "Нет белых пикселей. Невозможно определить радиус." << std::endl;
    }

    cv::Vec3d b_c_2d = {double(center.x), double(center.y), 1};
    cv::Vec3d e_c_1_2d = {double(center.x), double(center.y - mean_R), 1};
    cv::Vec3d e_c_2_2d = {double(center.x), double(center.y + mean_R), 1};

    cv::Mat b_c = K.inv() * b_c_2d;
    cv::Mat e_c_1 = K.inv() * e_c_1_2d;
    cv::Mat e_c_2 = K.inv() * e_c_2_2d;

    cv::Mat e_c = e_c_2 - e_c_1;

    cv::Mat b = ((ballRadius * 2) * b_c) / e_c.at<double>(0, 1);
    Vec3 ballCenter = {b.at<double>(0, 0), b.at<double>(0, 1), b.at<double>(0, 2)};
    return ballCenter;
}


std::vector<Point> getContourPoints(const cv::Mat& image, int n) {
    std::vector<Point> points;

    // 1. Найти координаты белых пикселей
    std::vector<Point> whitePixels;
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            if (image.at<uchar>(y, x) > 0) {
                whitePixels.push_back({x, y});
            }
        }
    }

    // 2. Найти центр кругов (приближенно - центр масс белых пикселей)
    Point center(0, 0);
    for (const auto& p : whitePixels) {
        center.x += p.x;
        center.y += p.y;
    }
    if (!whitePixels.empty()) {
        center.x /= whitePixels.size();
        center.y /= whitePixels.size();
    } else {
        std::cerr << "Нет белых пикселей. Невозможно найти контур." << std::endl;
        return points;
    }

    // 3. Определить радиус внешнего круга (приближенно - среднее расстояние от центра)
    double outerRadius = 0;
    for (const auto& p : whitePixels) {
        outerRadius += std::sqrt(std::pow(p.x - center.x, 2) + std::pow(p.y - center.y, 2));
    }
    if (!whitePixels.empty()) {
        outerRadius /= whitePixels.size();
    } else {
        std::cerr << "Нет белых пикселей. Невозможно определить радиус." << std::endl;
        return points;
    }

    // 4. Определить граничные пиксели внешнего круга (те, у которых есть черный сосед и расстояние до центра близко к радиусу)
    std::vector<Point> boundaryPixels;
    double radiusTolerance = 0.2 * outerRadius; // Погрешность для определения радиуса (20%)
    for (Point& p : whitePixels) {
        // Вычисляем расстояние от центра
        double distanceToCenter = std::sqrt(std::pow(p.x - center.x, 2) + std::pow(p.y - center.y, 2));

        // Проверяем, находится ли пиксель на границе и близок к радиусу внешнего круга
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

    // 5. Отсортировать граничные пиксели по углу относительно центра
    std::sort(boundaryPixels.begin(), boundaryPixels.end(), [&](const Point& a, const Point& b) {
        double angleA = std::atan2(a.y - center.y, a.x - center.x);
        double angleB = std::atan2(b.y - center.y, b.x - center.x);
        return angleA < angleB;
    });

    // 6. Выбрать n точек, равномерно распределенных по отсортированному списку
    if (boundaryPixels.size() < n) {
        std::cerr << "Недостаточно точек на контуре. Возвращаем все доступные точки." << std::endl;
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


Ball FindBall::estimate3dCoords_1(Ellipse const& ellipse, Camera const& camera) {
//    CameraIntrinsics intr;
//    intr.f  = camera.fx;    // пиксели
//    intr.cx = camera.cx;
//    intr.cy = camera.cy;
//    CameraExtrinsics extr;
//    computeCameraExtrinsics(0, 0, 0, 0, 0, 0, extr);
//    std::cout << "Матрица поворота R:" << std::endl;
//    for (int i = 0; i < 3; i++) {
//        std::cout << extr.R[i][0] << "\t"
//                  << extr.R[i][1] << "\t"
//                  << extr.R[i][2] << std::endl;
//    }
//    std::cout << "Вектор T: ("
//              << extr.T.x << ", "
//              << extr.T.y << ", "
//              << extr.T.z << ")" << std::endl;


    std::vector<cv::Point2f> edgeVectors2D = get2DPointsFromEllipse(ellipse, 10);
    std::vector<Vec3> edgeVectors;
    for (const auto& pixel : edgeVectors2D) {
        Vec3 ray = pixelToRay(pixel.x, pixel.y, camera.K);
        edgeVectors.push_back(ray);
    }
//    std::cout << "3d вектора:" << std::endl;
//    for (const auto& vec : edgeVectors) {
//        std::cout << "[" << vec.x << ", " << vec.y << ", " << vec.z << "]" << std::endl;
//    }

    Vec3 planeNormal = fitPlane(edgeVectors);
//    std::cout << "Нормаль: ("
//              << planeNormal.x << ", " << planeNormal.y << ", " << planeNormal.z << ")\n";

    double coneApexAngle = computeConeApexAngle(edgeVectors, planeNormal);
//    std::cout << "Угол при вершине конуса: " << coneApexAngle << std::endl;

    Vec3 Pc = computeSphereCenter(BALL_RADIUS, planeNormal, coneApexAngle);
//    std::cout << "Центр мяча в системе координат камеры: ("
//              << Pc.x << ", " << Pc.y << ", " << Pc.z << ")\n";

//    Vec3 Pw = cameraToWorld(Pc, extr);
//    std::cout << "Центр мяча в мировой системе координат: ("
//              << Pw.x << ", " << Pw.y << ", " << Pw.z << ")\n";

    Ball ball = {Pc.x, Pc.y, Pc.z};
    return ball;
}


Ball FindBall::estimate3dCoords_2(Ellipse const& ellipse, Camera const& camera) {
    Vec3 Pc = calculateBallCenter(ellipse, BALL_RADIUS, camera.get_camera_matrix());

    Ball ball = {Pc.x, Pc.y, Pc.z};
    return ball;
}

Ball FindBall::estimate3dCoords_3(Ellipse const& ellipse, Camera const& camera) {
    EdgeDetection edgeDetection(MIN_SIZE_OBJECT, MAX_SIZE_OBJECT);
    cv::Mat img_with_edgePoints = cv::Mat::zeros(cv::Size(camera.width, camera.height),CV_8UC1);
    for (int i = 0; i < edgePoints.size(); ++i) {
        edgePoints[i].x *= scale;
        edgePoints[i].y *= scale;
    }
    img_with_edgePoints = edgeDetection.draw_points(img_with_edgePoints, edgePoints);
    cv::imshow("edgeDetection", img_with_edgePoints);
    cv::waitKey(0);

    std::vector<Point> points = getContourPoints(img_with_edgePoints, 100);
    img_with_edgePoints = cv::Mat::zeros(cv::Size(camera.width, camera.height),CV_8UC1);
    img_with_edgePoints = edgeDetection.draw_points(img_with_edgePoints, points);
    cv::imshow("edgeDetection points", img_with_edgePoints);
    cv::waitKey(0);

    Vec3 Pc = calculateBallCenter(points, BALL_RADIUS, camera.get_camera_matrix());
    Ball ball = {Pc.x, Pc.y, Pc.z};
    return ball;
}


Ball FindBall::estimate3dCoords_4(Ellipse const& ellipse, Camera const& camera) {
    EdgeDetection edgeDetection(MIN_SIZE_OBJECT, MAX_SIZE_OBJECT);
    cv::Mat img_with_edgePoints = cv::Mat::zeros(cv::Size(camera.width, camera.height),CV_8UC1);
    for (int i = 0; i < edgePoints.size(); ++i) {
        edgePoints[i].x *= scale;
        edgePoints[i].y *= scale;
    }
    img_with_edgePoints = edgeDetection.draw_points(img_with_edgePoints, edgePoints);
//    cv::imshow("edgeDetection", img_with_edgePoints);
//    cv::waitKey(0);

    std::vector<Point> points = getContourPoints(img_with_edgePoints, 100);
    img_with_edgePoints = cv::Mat::zeros(cv::Size(camera.width, camera.height),CV_8UC1);
    img_with_edgePoints = edgeDetection.draw_points(img_with_edgePoints, points);
//    cv::imshow("edgeDetection points", img_with_edgePoints);
//    cv::waitKey(0);

    std::vector<Vec3> edgeVectors;
    for (const auto& pixel : points) {
        Vec3 ray = pixelToRay(pixel.x, pixel.y, camera.K);
        edgeVectors.push_back(ray);
    }
//    std::cout << "3d вектора:" << std::endl;
//    for (const auto& vec : edgeVectors) {
//        std::cout << "[" << vec.x << ", " << vec.y << ", " << vec.z << "]" << std::endl;
//    }
    Vec3 planeNormal = fitPlane(edgeVectors);
//    std::cout << "Нормаль: ("
//              << planeNormal.x << ", " << planeNormal.y << ", " << planeNormal.z << ")\n";
    double coneApexAngle = computeConeApexAngle(edgeVectors, planeNormal);
//    std::cout << "Угол при вершине конуса: " << coneApexAngle << std::endl;
    Vec3 Pc = computeSphereCenter(BALL_RADIUS, planeNormal, coneApexAngle);
    Ball ball = {Pc.x, Pc.y, Pc.z};
    return ball;
}
