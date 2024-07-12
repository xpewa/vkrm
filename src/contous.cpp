#include "contous.h"

cv::Mat detectEllipse1(cv::Mat image) {
    cv::Mat gray, edges;

//    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    cv::Canny(gray, edges, 50, 150);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(edges, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    for (size_t i = 0; i < contours.size(); i++) {
        if (contours[i].size() >= 5) {
            cv::RotatedRect ellipse = cv::fitEllipse(contours[i]);

            cv::ellipse(image, ellipse, cv::Scalar(0, 255, 0), 2);
        }
    }

    return image;
}


cv::Mat detectEllipse2(cv::Mat image, cv::Mat img_real) {
    threshold(image, image, 127, 255, cv::THRESH_BINARY); // бинаризация изображения

    cv::Mat sobelX, sobelY;
    Sobel(image, sobelX, CV_32F, 1, 0); // применение оператора Собеля по X
    Sobel(image, sobelY, CV_32F, 0, 1); // применение оператора Собеля по Y

    cv::Mat gradient;
    cv::sqrt(sobelX.mul(sobelX) + sobelY.mul(sobelY), gradient); // вычисление градиента

    std::vector<cv::Point> boundaryPoints;
    for (int i = 0; i < gradient.rows; i++) {
        for (int j = 0; j < gradient.cols; j++) {
            if (gradient.at<float>(i, j) > 0) { // если точка принадлежит границе объекта
                boundaryPoints.push_back(cv::Point(j, i));
            }
        }
    }

    cv::Mat covarianceMatrix = cv::Mat::zeros(2, 2, CV_32F);
    for (int i = 0; i < boundaryPoints.size(); i++) {
        covarianceMatrix.at<float>(0, 0) += boundaryPoints[i].x * boundaryPoints[i].x;
        covarianceMatrix.at<float>(0, 1) += boundaryPoints[i].x * boundaryPoints[i].y;
        covarianceMatrix.at<float>(1, 0) += boundaryPoints[i].y * boundaryPoints[i].x;
        covarianceMatrix.at<float>(1, 1) += boundaryPoints[i].y * boundaryPoints[i].y;
    }
    covarianceMatrix /= boundaryPoints.size();

    cv::Mat eigenvalues, eigenvectors;
    eigen(covarianceMatrix, eigenvalues, eigenvectors);

    float a = sqrt(eigenvalues.at<float>(0));
    float b = sqrt(eigenvalues.at<float>(1));
    float angle = atan2(eigenvectors.at<float>(1, 0), eigenvectors.at<float>(0, 0));

    std::cout << "Parameters of the ellipse:" << std::endl;
    std::cout << "a = " << a << std::endl;
    std::cout << "b = " << b << std::endl;
    std::cout << "angle = " << angle << std::endl;


    for (int i = 0; i < boundaryPoints.size(); i++) {
        img_real.at<cv::Vec3b>(boundaryPoints[i]) = cv::Vec3b(0,0,255);
    }

    cv::imshow("img_real", img_real);
    cv::waitKey(0);

    return image;
}

