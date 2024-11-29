cmake_minimum_required(VERSION 3.20)
project(EllipseRecognize)

add_subdirectory(EllipseRecognizeLibrary)

include(CTest)
add_subdirectory(TestsEllipseRecognizeLibrary)
enable_testing()