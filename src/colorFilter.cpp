#include "colorFilter.h"

cv::Mat ColorFilter::__getArrayFromData(cv::Mat const& img, cv::Mat const& mask){
    int Ny = img.rows;
    int Nx = img.cols;
    cv::Mat result = cv::Mat(0, 1, CV_32FC3);
    for (int y = 0; y < Ny; y++) {
        cv::Vec3b const * img_line = img.ptr<cv::Vec3b>(y);
        uchar const * mask_line = mask.ptr<uchar>(y);
        for (int x = 0; x < Nx; ++x) {
            if (mask_line[x] == 255)
                result.push_back(img_line[x]);
        }
    }
    result = result.reshape(1);
    result.convertTo(result, CV_32F);
    return result;
}

Cylinder ColorFilter::__train(cv::Mat const& pts) {
    cv::Mat data = __ransac(pts);

    Cylinder f;
    cv::Scalar mean_scalar_0 = cv::mean(data.col(0));
    cv::Scalar mean_scalar_1 = cv::mean(data.col(1));
    cv::Scalar mean_scalar_2 = cv::mean(data.col(2));
    cv::Mat mean(1, 3, CV_32F);
    mean.at<float>(0) = mean_scalar_0[0];
    mean.at<float>(1) = mean_scalar_1[0];
    mean.at<float>(2) = mean_scalar_2[0];
    f.p0 = mean;
    mean = cv::Mat::ones(data.rows, 1, CV_32F) * mean;

    cv::Mat p_p0 = data - mean;
    cv::SVD svd(p_p0);
    f.v = svd.vt.row(0);

    cv::Mat t = (p_p0) * f.v.t();
    cv::sort(t, t, cv::SORT_EVERY_COLUMN);

    f.t1 = t.at<float>(round(t.rows * 0.05), 0);
    f.t2 = t.at<float>(round((t.rows - 1) * 0.95), 0);

    cv::Mat A = t * f.v + mean - data;
    cv::Mat dp;
    sqrt(A.mul(A)*cv::Mat::ones(3, 1, CV_32F), dp);
    cv::sort(dp, dp, cv::SORT_EVERY_COLUMN);
    f.R = dp.at<float>(round((dp.rows - 1) * 0.4), 0); // radius
    std::cout << "R " << f.R << std::endl;

    return f;
}

cv::Mat ColorFilter::recognize(int countDatasetImg, cv::Mat const& img) {
    Cylinder f = __getCylinder(countDatasetImg);
    int Ny = img.rows;
    int Nx = img.cols;
    cv::Mat mask_img(img.rows, img.cols, CV_8U, cv::Scalar(255));
    cv::Mat p = __getArrayFromData(img, mask_img);

    cv::Mat mean = cv::Mat::ones(p.rows, 1, CV_32F) * f.p0;
    cv::Mat p_p0 = p - mean;
    cv::Mat t = (p_p0) * f.v.t();

    cv::Mat dt = abs(t - (f.t1 + f.t2) / 2) - (f.t2 - f.t1) / 2;
    dt = cv::max(dt, 0);

    cv::Mat A = t * f.v + mean - p;
    cv::Mat dp;
    sqrt(A.mul(A)*cv::Mat::ones(3, 1, CV_32F), dp);
//    float R = dp.at<float>(round((dp.rows - 1) * 0.5), 0);
    dp = cv::max(dp - f.R * cv::Mat::ones(dp.rows, 1, CV_32F), 0);

    cv::Mat d;
    d = dp + dt;
    d = d.reshape(1, Ny);
    d.convertTo(d, CV_8U);

    cv::Mat mask = cv::Mat(Ny, Nx, CV_8U, cv::Scalar(0));
    for (int y = 0; y < Ny; y++) {
        uchar * line = d.ptr<uchar>(y);
        uchar * mask_line = mask.ptr<uchar>(y);
        for (int x = 0; x < Nx; x++) {
            if (line[x] == 0) {
                mask_line[x] = 255;
            }
        }
    }
    return mask;
}

Cylinder ColorFilter::__getCylinder(int countImg) {
    ColorFilter colorFilter;
    std::vector<cv::Mat> img(countImg + 1);
    std::vector<cv::Mat> img_mask(countImg + 1);

    for (int i = 1; i < countImg + 1; ++i) {
        img[i] = cv::imread("../img_color/IMG_" + std::to_string(i) + ".jpg");
        img_mask[i] = cv::imread("../img_color/IMG_" + std::to_string(i) + ".png",cv::IMREAD_GRAYSCALE);

        resize(img[i], img[i], cv::Size(500, 500), cv::INTER_LINEAR);
        resize(img_mask[i], img_mask[i], cv::Size(500, 500), cv::INTER_LINEAR);
    }

    std::vector<cv::Mat> data(countImg + 1);

    for (int i = 1; i < countImg + 1; ++i) {
        data[i] = __getArrayFromData(img[i], img_mask[i]);
    }

    cv::Mat data_train;

    for (int i = 1; i < countImg + 1; ++i) {
        data_train.push_back(data[i]);
    }

    return __train(data_train);
}


double ColorFilter::__calculateDistancePointPoint(cv::Vec3b const& p1, cv::Vec3b const& p2) {
    return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2) + pow(p1[2] - p2[2], 2));
}

double ColorFilter::__calculateDistancePointLine(cv::Vec3b const& pl1, cv::Vec3b const& pl2, cv::Vec3b const& point) {
    int x1 = point[0];
    int y1 = point[1];
    int z1 = point[2];
    int x0 = pl1[0];
    int y0 = pl1[1];
    int z0 = pl1[2];
    int q = pl2[0] - pl1[0];
    int r = pl2[1] - pl1[1];
    int s = pl2[2] - pl1[2];

    if (sqrt(pow(q, 2) + pow(r, 2) + pow(s, 2)))
        return (pow(r*(z1-z0) - s*(y1-y0), 2) + pow(q*(z1-z0) - s*(x1-x0), 2) + pow(q*(y1-y0) - r*(x1-x0), 2)) / (sqrt(pow(q, 2) + pow(r, 2) + pow(s, 2)));
    else
        return 0; // ?
}


cv::Mat ColorFilter::__ransac(cv::Mat const& pts) {
    srand(time(0));
    int count_pts = pts.rows;

    cv::Mat bestInliers = cv::Mat(0, 1, CV_32FC3);
    int maxIterations = 100;
    double threshold = 70000.0;

    for (int i = 0; i < maxIterations; ++i) {
        // Выбираем две случайные точки
        int index1 = rand() % (count_pts);
        int index2 = rand() % (count_pts);
        cv::Vec3b p1 = pts.row(index1);
        cv::Vec3b p2 = pts.row(index2);

        cv::Mat inliers = cv::Mat(0, 3, CV_32F);
        for (int i = 0; i < count_pts; ++i) {
            cv::Vec3b p = pts.row(i);
            double distance = __calculateDistancePointLine(p1, p2, p);
            if (distance < threshold) {
                cv::Mat temp(1, 3, CV_32F);
                temp.col(0) = p[0];
                temp.col(1) = p[1];
                temp.col(2) = p[2];
                inliers.push_back(temp);
            }
        }

        // Если найдено больше инлаеров, чем в предыдущей итерации
        if (inliers.rows > bestInliers.rows) {
            bestInliers = inliers;
        }
        if (bestInliers.rows > count_pts * 0.9) {
            break;
        }
    }
    return bestInliers;
}
