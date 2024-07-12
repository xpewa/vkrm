cmake_minimum_required(VERSION 3.20)
project(vkrm)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_STANDARD 17)
add_executable(vkrm
        src/main.cpp
        src/colorFilter.cpp
        src/contous.cpp
        src/edgeDetection.cpp
        src/edgeDetection.cpp
        src/detectEllipse.cpp)
target_link_libraries(vkrm ${OpenCV_LIBS})
target_include_directories(vkrm PUBLIC inc/)
