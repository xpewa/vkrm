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


struct Contour {
    std::vector<Point> pixels;
};


class EdgeDetection {
    int min_size_object;
    int max_size_object;
public:
    std::vector<Contour> __findContours(cv::Mat const & image);
    Contour __filterMaxContour(std::vector<Contour> & objects);
    std::vector<Contour> __filterSize(std::vector<Contour> & objects);
    std::vector<Contour> __filterCircularity(std::vector<Contour> & objects);
    cv::Mat __GaussianBlur(cv::Mat const & img);
    std::vector<Point> __PrevittOperator(cv::Mat const & img);
    cv::Mat __fillBlank(cv::Mat const & img);
    std::vector<Point> find_points(cv::Mat const & src);
    cv::Mat draw_points(cv::Mat const & img, std::vector<Point> const & points);
    std::vector<Point> __PrevittOperatorOptimized(const cv::Mat& img);

    EdgeDetection(int min_size_object = 100, int max_size_object = 100000000)
    : min_size_object(min_size_object), max_size_object(max_size_object) {}
};

#endif //VKRM_EDGEDETECTION_H
