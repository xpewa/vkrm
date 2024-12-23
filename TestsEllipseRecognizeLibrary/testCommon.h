#ifndef ELLIPSERECOGNIZE_TESTCOMMON_H
#define ELLIPSERECOGNIZE_TESTCOMMON_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cassert>
#include <fstream>
#include <string>
#include <numeric>
#include <chrono>
#include <vector>
#include <numeric>

#include "colorFilter.h"
#include "edgeDetection.h"
#include "detectEllipse.h"
#include "findBall.h"

//std::string PATH_CYLINDER = "../../cylinder.txt";

std::vector<Ellipse> readCentersFromFile(const std::string& filename);

#endif //ELLIPSERECOGNIZE_TESTCOMMON_H
