#include "videoRecognize.h"

//cv::Mat VideoRecognize::read_next_part_of_video() {
//    return cv::Mat();
//}

void VideoRecognize::recognize_ellipse_in_video() {
    int frame_width = this->cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = this->cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    this->video = cv::VideoWriter("../../videos/outVideo.mp4", cv::VideoWriter::fourcc('m','p','4','v'), 30, cv::Size(frame_width,frame_height));

    while (true) {
        cv::Mat frame;
        this->cap >> frame;
        if (frame.empty())
            break;

        cv::Mat img_new = colorFilter.recognize(frame);

        EdgeDetection edgeDetection;
        std::vector<Point> imagePoints = edgeDetection.find_points(img_new);
        cv::Mat emptyImg = cv::Mat::zeros(cv::Size(img_new.cols, img_new.rows),CV_8UC1);
        emptyImg = edgeDetection.draw_points(emptyImg, imagePoints);
//        cv::imshow("edgeDetection", emptyImg);
//        cv::waitKey(0);

        DetectEllipse detectEllipse;
        Ellipse ellipse = detectEllipse.detectEllipse(imagePoints);
//        std::cout << "Ellipse center: (" << ellipse.x << ", " << ellipse.y << ")" << std::endl;
        cv::Point centerCircle1(ellipse.x, ellipse.y);
        cv::Scalar colorCircle1(0, 0, 255);
        cv::circle(frame, centerCircle1, 10, colorCircle1, cv::FILLED);
//        cv::imshow("img res", frame);
//        cv::waitKey(0);

        video.write(frame);
    }
    return;
}
