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
//    cv::Mat emptyImg = cv::Mat::zeros(cv::Size(img_res.cols, img_res.rows),CV_8UC1);
//    emptyImg = edgeDetection.draw_points(emptyImg, imagePoints);
//    cv::imshow("edgeDetection", emptyImg);
//    cv::waitKey(0);
    DetectEllipse detectEllipse;
    Ellipse ellipse = detectEllipse.detectEllipse(imagePoints);

    ellipse.x *= scale;
    ellipse.y *= scale;
    ellipse.R1 *= scale;
    ellipse.R2 *= scale;

    return ellipse;
}


cv::Vec3d findPlaneNormal(const std::vector<cv::Vec3d>& vectors) {
    cv::Vec3d sum_of_vectors(0, 0, 0);
    for (const auto& vec : vectors) {
        double magnitude = sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
        if (magnitude == 0) continue;
        sum_of_vectors += vec / magnitude;
    }

    cv::Vec3d average_vector = sum_of_vectors / static_cast<double>(vectors.size());

    double average_magnitude = sqrt(average_vector[0] * average_vector[0] +
                                    average_vector[1] * average_vector[1] +
                                    average_vector[2] * average_vector[2]);

    if (average_magnitude == 0)
        return cv::Vec3d(0,0,0);

    return average_vector / average_magnitude;
}

double findConeApexAngle(const std::vector<cv::Vec3d>& points, const cv::Vec3d& normal) {
    double sumAngles = 0.0;
    double normalMagnitude = sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);

    for (const auto& point : points) {
        double dotProduct = point[0] * normal[0] + point[1] * normal[1] + point[2] * normal[2];
        double pointMagnitude = sqrt(point[0]*point[0] + point[1]*point[1] + point[2]*point[2]);

        double cosAngle = dotProduct / (normalMagnitude * pointMagnitude);

        if (abs(cosAngle) > 1.0) {
            cosAngle = (cosAngle > 0) ? 1.0 : -1.0;
        }

        sumAngles += acos(cosAngle);
    }
    return 2.0 * sumAngles / points.size(); // в радианах
}

// Функция для вычисления 3D координат мяча
cv::Vec3d findBallCoordinates(const cv::Vec3d& normal, double coneAngle, double ballRadius) {
    double sinHalfAngle = sin(coneAngle / 2.0);
    double normalMagnitude = sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);
    double distance = ballRadius / sinHalfAngle;

    std::cout << "distance: " << distance << std::endl;

    return (distance / normalMagnitude) * normal;
}

// Функция для создания набора 3D точек из эллипса на изображении
std::vector<cv::Vec3d> get3DPointsFromEllipse(const Camera& camera, const Ellipse& ellipse, int numPoints) {
    std::vector<cv::Vec3d> points3D;

    std::vector<cv::Point2f> ellipsePoints;
    for (int i = 0; i < numPoints; ++i) {
        double angle = 2 * M_PI * i / numPoints;
        float x = ellipse.R1 * cos(angle);
        float y = ellipse.R2 * sin(angle);

        // относительно нуля
        cv::Mat rotationMatrix = cv::getRotationMatrix2D(cv::Point2f(0,0), ellipse.angle, 1.0);

        // относительно нуля
        cv::Mat point2D = (cv::Mat_<double>(3, 1) << x, y, 1);

        cv::Mat rotatedPoint = rotationMatrix * point2D;

        // Получаем точку после поворота, и смещаем ее.
        ellipsePoints.push_back(cv::Point2f(rotatedPoint.at<double>(0) + ellipse.x, rotatedPoint.at<double>(1) + ellipse.y));
    }
//    cv::Mat new_img = cv::Mat::zeros(cv::Size(1280, 1024),CV_8UC1);
//    for (int i = 0; i < ellipsePoints.size(); ++i) {
//        cv::Point2f p = ellipsePoints[i];
//        cv::Point centerCircle(p.x, p.y);
//        cv::Scalar colorCircle(255);
//        cv::circle(new_img, centerCircle, 1, colorCircle, cv::FILLED);
//    }
//    cv::imshow("ellipsePoints", new_img);
//    cv::waitKey(0);

    // 2. Обратная проекция каждой точки в 3D
    for (const auto& point2D : ellipsePoints) {
        std::vector<cv::Point2f> distorted_points = {point2D};
        std::vector<cv::Point2f> undistorted_points;
        cv::undistortPoints(distorted_points, undistorted_points, camera.get_camera_matrix(), camera.get_distortion_coeff());
        cv::Point2f undistorted_point = undistorted_points[0];
//        cv::Point2f undistorted_point = point2D;

        cv::Mat point3DHomogeneous = (cv::Mat_<double>(3,1) << undistorted_point.x, undistorted_point.y, 1);

        points3D.push_back(cv::Vec3d(point3DHomogeneous.at<double>(0),
                                     point3DHomogeneous.at<double>(1),
                                     point3DHomogeneous.at<double>(2)));
    }
    return points3D;
}

cv::Vec3d transformVectorCameraToWorld(const Camera& camera, const cv::Vec3d& vector) {
    cv::Mat vector_homogeneous = (cv::Mat_<double>(4, 1) << vector[0], vector[1], vector[2], 1);

    std::cout << camera.get_world_to_camera_matrix() << std::endl;

    cv::Mat camera_to_world_matrix = camera.get_world_to_camera_matrix().inv(cv::DECOMP_SVD);
    cv::Mat camera_to_world =  (cv::Mat_<double>(4, 4) <<
            0.6947, -0.3040, 0.6519, 2.0000,
            0.7193,  0.2936, -0.6296, -2.0000,
            0.0000,  0.9063,  0.4226,  1.0000,
            0.0000,  0.0000,  0.0000,  1.0000);

    cv::Mat world_vector_homogeneous = camera_to_world * vector_homogeneous;
    cv::Vec3d world_vector(world_vector_homogeneous.at<double>(0),
                       world_vector_homogeneous.at<double>(1),
                       world_vector_homogeneous.at<double>(2));
    return world_vector;
}


Ball FindBall::estimate3dCoords(Ellipse ellipse, Camera camera) {

    std::vector<cv::Vec3d> points = get3DPointsFromEllipse(camera, ellipse, 10);

    cv::Vec3d normal = findPlaneNormal(points);
    std::cout << "normal: " << normal << std::endl;

    double coneAngle = findConeApexAngle(points, normal);
    std::cout << "Cone Apex Angle: " << coneAngle * 180 / M_PI << " degrees" << std::endl;

    cv::Vec3d ballCoordinates = findBallCoordinates(normal, coneAngle, BALL_RADIUS);
//    ballCoordinates = transformVectorCameraToWorld(camera, ballCoordinates);

    Ball ball;
    ball.x = ballCoordinates[0];
    ball.y = ballCoordinates[1];
    ball.z = ballCoordinates[2];
    return ball;
}


Ball FindBall::findBall(cv::Mat const & img) {
    Ellipse ellipse = getEllipseParameters(img);

    Camera camera;

    Ball ball = estimate3dCoords(ellipse, camera);

    return ball;
}