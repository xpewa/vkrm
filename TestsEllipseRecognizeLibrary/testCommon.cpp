#include "testCommon.h"


std::vector<Ellipse> readCentersFromFile(const std::string& filename) {
    std::vector<Ellipse> ellipses;
    std::ifstream file(filename);

    if (file.is_open()) {
        std::string line;
        while (getline(file, line)) {
            std::stringstream ss(line);
            Ellipse ellipse;
            ss >> ellipse.x >> ellipse.y;
            if (ss) {
                ellipses.push_back(ellipse);
            } else {
                std::cerr << "Error reading line: " << line << std::endl;
            }
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
    return ellipses;
}
