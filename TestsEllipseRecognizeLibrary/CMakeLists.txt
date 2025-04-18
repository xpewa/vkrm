cmake_minimum_required(VERSION 3.20)
project(VKRMTests)

find_package(OpenCV REQUIRED)
set(CMAKE_CXX_STANDARD 17)

add_executable(testsColorFilter testsColorFilter.cpp testCommon.cpp)
target_link_libraries(testsColorFilter vkrm_lib)

add_executable(testsDetectEllipse testDetectEllipse.cpp testCommon.cpp)
target_link_libraries(testsDetectEllipse vkrm_lib)

add_executable(testFindBall testFindBall.cpp testCommon.cpp)
target_link_libraries(testFindBall vkrm_lib)

add_test(NAME testsColorFilter COMMAND testsColorFilter)
add_test(NAME testsDetectEllipse COMMAND testsDetectEllipse)
add_test(NAME testFindBall COMMAND testFindBall)