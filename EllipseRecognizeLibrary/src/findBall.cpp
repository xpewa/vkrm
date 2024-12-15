#include "findBall.h"


Ellipse FindBall::findBall(cv::Mat const & img) {
    cv::Mat resize_img = img.clone();
    int img_width = img.cols;
    int img_height = img.rows;
    resize(img, resize_img, cv::Size(img_width / scale, img_height / scale), cv::INTER_LINEAR);


    cv::Mat img_res = colorFilter.recognize(resize_img);
    EdgeDetection edgeDetection;
    std::vector<Point> imagePoints = edgeDetection.find_points(img_res);
    DetectEllipse detectEllipse;
    Ellipse ellipse = detectEllipse.detectEllipse(imagePoints);

    ellipse.x *= scale;
    ellipse.y *= scale;

    return ellipse;
}