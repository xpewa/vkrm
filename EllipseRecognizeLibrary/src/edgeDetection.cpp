#include "edgeDetection.h"


cv::Mat EdgeDetection::__GaussianBlur(cv::Mat const & img) {
    cv::Mat res(cv::Size(img.cols, img.rows), CV_8UC1, 255);
    for (int y = 1; y < img.rows - 1; ++y) {
        for (int x = 1; x < img.cols - 1; ++x) {
            float k1 = 0.0625;
            float k2 = 0.125;
            float k3 = 0.0625;
            float k4 = 0.125;
            float k5 = 0.25;
            float k6 = 0.125;
            float k7 = 0.0625;
            float k8 = 0.125;
            float k9 = 0.0625;

            int p1 = img.at<uchar>(y - 1, x - 1);
            int p2 = img.at<uchar>(y - 1, x);
            int p3 = img.at<uchar>(y - 1, x + 1);
            int p4 = img.at<uchar>(y, x - 1);
            int p5 = img.at<uchar>(y, x);
            int p6 = img.at<uchar>(y, x + 1);
            int p7 = img.at<uchar>(y + 1, x - 1);
            int p8 = img.at<uchar>(y + 1, x);
            int p9 = img.at<uchar>(y + 1, x + 1);

            res.at<uchar>(y, x) = k1*p1 + k2*p2 + k3*p3 + k4*p4 + k5*p5 + k6*p6 + k7*p7 + k8*p8 + k9*p9;
        }
    }
    return res;
}

std::vector<Point> EdgeDetection::__PrevittOperator(cv::Mat const & img) {

    std::vector<std::vector<int>> Gx(img.cols, std::vector<int>(img.rows, 0));
    std::vector<std::vector<int>> Gy(img.cols, std::vector<int>(img.rows, 0));
    std::vector<std::vector<int>> Hp(img.cols, std::vector<int>(img.rows, 0));
    std::vector<Point> points;

    for (int y = 1; y < img.rows - 1; ++y) {
        const unsigned char* row_prev = img.ptr<uchar>(y - 1);
        const unsigned char* row_current = img.ptr<uchar>(y);
        const unsigned char* row_next = img.ptr<uchar>(y + 1);
        for (int x = 1; x < img.cols - 1; ++x) {
            const unsigned char z1 = row_prev[x - 1];
            const unsigned char z2 = row_prev[x];
            const unsigned char z3 = row_prev[x + 1];
            const unsigned char z4 = row_current[x - 1];
            const unsigned char z6 = row_current[x + 1];
            const unsigned char z7 = row_next[x - 1];
            const unsigned char z8 = row_next[x];
            const unsigned char z9 = row_next[x + 1];

            int gx = (z7 + z8 + z9) - (z1 + z2 + z3);
            int gy = (z3 + z6 + z9) - (z1 + z4 + z7);
            Gx[x][y] = gx;
            Gy[x][y] = gy;
        }
    }

    int rows = Gx.size();
    int cols = Gx[0].size();
    float k = 0.2; // 0.2

    for (int x = 10; x < rows - 10; ++x) {
        for (int y = 10; y < cols - 10; ++y) {
            int gp1 = 0;
            int gp2 = 0;
            int gp3 = 0;

            for (int i = x - 1; i < x + 2; ++i) {
                for (int j = y - 1; j < y + 2; ++j) {
                    int gx = Gx[i][j];
                    int gy = Gy[i][j];
                    gp1 += gx*gx;
                    gp2 += gx*gy;
                    gp3 += gy*gy;
                }
            }

            int hp = (gp1 * gp3 - gp2*gp2) - k*(gp1 + gp3)*(gp1 + gp3);
            Hp[x][y] = hp;

            if (hp > 10000000 || hp < -10000000) {
                points.emplace_back(x, y);
            }
        }
    }

    return points;
}


cv::Mat EdgeDetection::__fillBlank(cv::Mat const & img) {
    int sizeRange1 = 3;
    int sizeRange2 = 3;
    cv::Mat new_img2 = img.clone();
    for (int y = sizeRange2; y < img.rows - sizeRange2 - 1; ++y) {
        for (int x = sizeRange2; x < img.cols - sizeRange2 - 1; ++x) {
            if (img.at<uchar>(y, x) == 0) {
                for (int y0 = y - sizeRange2; y0 < y + sizeRange2; ++y0) {
                    for (int x0 = x - sizeRange2; x0 < x + sizeRange2; ++x0) {
                        if (img.at<uchar>(y0, x0) == 0) {
                            new_img2.at<uchar>(y, x) = 0;
                            break;
                        }
                    }
                }
            }
        }
    }
    cv::Mat new_img1 = new_img2.clone();
    for (int y = sizeRange1; y < img.rows - sizeRange1 - 1; ++y) {
        for (int x = sizeRange1; x < img.cols - sizeRange1 - 1; ++x) {
            if (img.at<uchar>(y, x) == 0) {
                for (int y0 = y - sizeRange1; y0 < y + sizeRange1; ++y0) {
                    for (int x0 = x - sizeRange1; x0 < x + sizeRange1; ++x0) {
                        if (new_img2.at<uchar>(y0, x0) == 255) {
                            new_img1.at<uchar>(y, x) = 255;
                            break;
                        }
                    }
                }
            }
        }
    }
    return new_img1;
}


bool isBorderPixel(const cv::Mat& image, int x, int y) {
    int height = image.rows;
    int width = image.cols;
    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) {
        return false;
    }
    if (image.at<uchar>(y, x) == 0) {
        return false;
    }
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            if (image.at<uchar>(y + dy, x + dx) == 0) {
                return true;
            }
        }
    }
    return false;
}


std::vector<Point> findExternalBorder(const cv::Mat& image) {
    std::vector<Point> borderImage;
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            if (isBorderPixel(image, x, y)) {
                borderImage.push_back(Point(x, y));
            }
        }
    }
    return borderImage;
}


std::vector<Point> EdgeDetection::find_points(cv::Mat const & src) {
    cv::Mat img = src.clone();

    std::vector<Contour> contours = __findContours(img);

    cv::Mat new_img = cv::Mat::zeros(cv::Size(img.cols, img.rows),CV_8UC1);
    img = draw_points(new_img, contours[0].pixels);
    for (Contour& countor : contours) {
        img = draw_points(img, countor.pixels);
    }
//    cv::imshow("__findContours initial", img);
//    cv::waitKey(0);

    contours = __filterSize(contours);

    new_img = cv::Mat::zeros(cv::Size(img.cols, img.rows),CV_8UC1);
    img = draw_points(new_img, contours[0].pixels);
    for (Contour& countor : contours) {
        img = draw_points(img, countor.pixels);
    }
//    cv::imshow("__filterSize", img);
//    cv::waitKey(0);

    contours = __filterCircularity(contours);

    new_img = cv::Mat::zeros(cv::Size(img.cols, img.rows),CV_8UC1);
    img = draw_points(new_img, contours[0].pixels);
    for (Contour& countor : contours) {
        img = draw_points(img, countor.pixels);
    }
//    cv::imshow("__filterCircularity", img);
//    cv::waitKey(0);

    Contour max_contour = __filterMaxContour(contours);

    cv::Mat clear_img = cv::Mat::zeros(cv::Size(img.cols, img.rows),CV_8UC1);
    img = draw_points(clear_img, max_contour.pixels);
//    cv::imshow("max_contour", img);
//    cv::waitKey(0);

//    std::vector<Point> points = __PrevittOperator(img);
    std::vector<Point> points;
    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            if (img.at<uchar>(y, x) != 0) {
                points.push_back(Point(x, y));
            }
        }
    }
    clear_img = cv::Mat::zeros(cv::Size(img.cols, img.rows),CV_8UC1);
    img = draw_points(clear_img, points);
    points = findExternalBorder(img);

    return points;
}


cv::Mat EdgeDetection::draw_points(cv::Mat const & img, std::vector<Point> const & points) {
    cv::Mat res = img.clone();
    for (int i = 0; i < points.size(); ++i) {
        Point p = points[i];
        cv::Point centerCircle(p.x, p.y);
        cv::Scalar colorCircle(255);
        cv::circle(res, centerCircle, 1, colorCircle, cv::FILLED);
    }
    return res;
}


std::vector<Contour> EdgeDetection::__findContours(cv::Mat const & image) {
    int threshold = 127;
    std::vector<Contour> contours;
    int height = image.rows;
    int width = image.cols;
    std::vector<std::vector<bool>> visited(height, std::vector<bool>(width, false));

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (image.at<uchar>(y, x) > threshold && !visited[y][x]) {
                Contour contour;
                std::queue<Point> q;
                q.push(Point(x, y));
                visited[y][x] = true;

                while (!q.empty()) {
                    Point current = q.front();
                    q.pop();
                    contour.pixels.push_back(current);

                    int dx[] = {-1, 1, 0, 0};
                    int dy[] = {0, 0, -1, 1};

//                    int dx[] = {-1, 1, 0, 0, -1, -1, 1, 1}; // 8-связность
//                    int dy[] = {0, 0, -1, 1, -1, 1, -1, 1};

                    for (int i = 0; i < 4; ++i) {
                        int nextX = current.x + dx[i];
                        int nextY = current.y + dy[i];
                        if (nextX >= 0 && nextX < width && nextY >= 0 && nextY < height &&
                            image.at<uchar>(nextY, nextX) > threshold && !visited[nextY][nextX]) {
                            q.push(Point(nextX, nextY));
                            visited[nextY][nextX] = true;
                        }
                    }
                }
                contours.push_back(contour);
            }
        }
    }
    return contours;
}


Contour EdgeDetection::__filterMaxContour(std::vector<Contour> & objects) {
    if (objects.size() == 0) {
        Contour contour;
        return contour;
    }
    uint max_size = 0;
    int max_size_i = 0;
    for (int i = 0; i < objects.size(); ++i) {
        if (objects[i].pixels.size() > max_size) {
            max_size = objects[i].pixels.size();
            max_size_i = i;
        }
    }
    return objects[max_size_i];
}


std::vector<Contour> EdgeDetection::__filterSize(std::vector<Contour> & objects) {
    if (objects.size() == 0) return objects;
    std::vector<Contour> filteredContours;
    for (Contour& object : objects) {
        double area = 0;
        for (Point& point : object.pixels) {
            area += 1;
        }
        if (area > this->min_size_object and area < this->max_size_object) {
            filteredContours.push_back(object);
        }
    }
    return filteredContours;
}


std::vector<Contour> EdgeDetection::__filterCircularity(std::vector<Contour> & objects) {
    if (objects.size() == 0) return objects;
    int maxRatio = 2;
    std::vector<Contour> filteredContours;
    for (Contour& object : objects) {
        int min_x = 1000000000;
        int min_y = 1000000000;
        int max_x = 0;
        int max_y = 0;
        for (Point& point : object.pixels) {
            if (point.x < min_x) min_x = point.x;
            if (point.y < min_y) min_y = point.y;
            if (point.x > max_x) max_x = point.x;
            if (point.y > max_y) max_y = point.y;
        }
        int dx = max_x - min_x;
        int dy = max_y - min_y;
        if (dx == 0 || dy == 0) continue;
        double circularity = std::max(dx, dy) / std::min(dx, dy);
        if (circularity < maxRatio) {
            filteredContours.push_back(object);
        }
    }
    return filteredContours;
}
