#ifndef VKRM_EDGEDETECTION_H
#define VKRM_EDGEDETECTION_H

#include <iostream>
#include <opencv2/opencv.hpp>

struct Point {
    Point (int x, int y) : x(x), y(y) {}
    Point () {}
    int x;
    int y;

    friend std::ostream& operator<<(std::ostream& out, const Point& p){
        return out << "Point: " << p.x << " " << p.y;
    }
    static bool comp(Point p1, Point p2) {
        int epsilon = 20;
        return abs(p1.x - p2.x) > epsilon ? p1.x < p2.x : p1.y < p2.y;
    }
    static void sortPoint(std::vector<Point> & points) {
        std::sort(points.begin(), points.end(), Point::comp);
    }
};

class EdgeDetection {
public:
    cv::Mat __GaussianBlur(cv::Mat const & img);
    std::vector<Point> __PrevittOperator(cv::Mat const & img);
    cv::Mat __fillBlank(cv::Mat const & img);
    std::vector<Point> find_points(cv::Mat const & src);
    cv::Mat draw_points(cv::Mat const & img, std::vector<Point> const & points);
};

#endif //VKRM_EDGEDETECTION_H
