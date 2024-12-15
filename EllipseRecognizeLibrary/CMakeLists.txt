cmake_minimum_required(VERSION 3.20)
project(vkrm)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_STANDARD 17)

set( vkrm_source
        src/main.cpp
        src/colorFilter.cpp
        src/contous.cpp
        src/edgeDetection.cpp
        src/edgeDetection.cpp
        src/detectEllipse.cpp
        src/videoRecognize.cpp
        src/findBall.cpp

        inc/colorFilter.h
        inc/contous.h
        inc/edgeDetection.h
        inc/edgeDetection.h
        inc/detectEllipse.h
        inc/videoRecognize.h
        inc/findBall.h
)

add_executable(vkrm ${vkrm_source})
target_link_libraries(vkrm ${OpenCV_LIBS})
target_include_directories(vkrm PUBLIC inc)

add_library( vkrm_lib STATIC ${vkrm_source} )
target_link_libraries(vkrm_lib ${OpenCV_LIBS})
target_include_directories(vkrm_lib PUBLIC inc)
