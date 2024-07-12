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
    cv::Mat res(cv::Size(img.cols, img.rows), CV_8UC1, 255);
    for (int y = 1; y < img.rows - 1; ++y) {
        for (int x = 1; x < img.cols - 1; ++x) {
            res.at<uchar>(y, x) = img.at<uchar>(y, x);
        }
    }

    std::vector<std::vector<int>> Gx(img.cols, std::vector<int>(img.rows, 0));
    std::vector<std::vector<int>> Gy(img.cols, std::vector<int>(img.rows, 0));
    std::vector<std::vector<int>> Hp(img.cols, std::vector<int>(img.rows, 0));
    std::vector<Point> points;

    for (int y = 1; y < img.rows - 1; ++y) {
        for (int x = 1; x < img.cols - 1; ++x) {
            int z1 = img.at<uchar>(y - 1, x - 1);
            int z2 = img.at<uchar>(y - 1, x);
            int z3 = img.at<uchar>(y - 1, x + 1);
            int z4 = img.at<uchar>(y, x - 1);
            int z6 = img.at<uchar>(y, x + 1);
            int z7 = img.at<uchar>(y + 1, x - 1);
            int z8 = img.at<uchar>(y + 1, x);
            int z9 = img.at<uchar>(y + 1, x + 1);

            int gx = (z7 + z8 + z9) - (z1 + z2 + z3);
            int gy = (z3 + z6 + z9) - (z1 + z4 + z7);
            Gx[x][y] = gx;
            Gy[x][y] = gy;
        }
    }

    for (int x = 10; x < Gx.size() - 10; ++x) {
        for (int y = 10; y < Gx[0].size() - 10; ++y) {
            float k = 0.2; // 0.2
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
                Point p = Point(x, y);
                points.push_back(p);
            }
        }
    }

    return points;
}

cv::Mat EdgeDetection::__fillBlank(cv::Mat const & img) {
    int sizeRange1 = 10;
    int sizeRange2 = 10;
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

std::vector<Point> EdgeDetection::find_points(cv::Mat const & src) {
    cv::Mat img = src.clone();

//    img = __GaussianBlur(img);
//    for (int i = 0; i < 10; ++i) {
//        img = __GaussianBlur(img);
//    }

    img = __fillBlank(img);

    std::vector<Point> points = __PrevittOperator(img);

    return points;
}

cv::Mat EdgeDetection::draw_points(cv::Mat const & img, std::vector<Point> const & points) {
    uint count_points = 0;
    for (int i = 0; i < points.size(); ++i) {
        ++count_points;
    }
    std::cout << "Count points: " << count_points << std::endl;

    cv::Mat res = img.clone();
    for (int i = 0; i < points.size(); ++i) {
        Point p = points[i];
        cv::Point centerCircle(p.x, p.y);
        cv::Scalar colorCircle(255);
        cv::circle(res, centerCircle, 1, colorCircle, cv::FILLED);
    }
    return res;
}
