################################################################################
# Check required CMake version
set(REQUIRED_CMAKE_VERSION "3.18.0")
cmake_minimum_required(VERSION ${REQUIRED_CMAKE_VERSION})
set(CMAKE_POLICY_VERSION_MINIMUM ${REQUIRED_CMAKE_VERSION})
if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.0.0")
    set(POLYSOLVE_WITH_AMGCL OFF CACHE BOOL "Use AMGCL for solving linear systems")
endif()

project(polysolve_python LANGUAGES CXX)
################################################################################

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

if(NOT DEFINED CPM_SOURCE_CACHE)
    set(CPM_SOURCE_CACHE "${CMAKE_BINARY_DIR}/cpm-cache" CACHE PATH "CPM source cache")
endif()


# Sort projects inside the solution
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Generate position independent code by default
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

################################################################################
# Polysolve library
################################################################################

# dependencies
include(polysolve)
include(pybind11)
include(pybind11_json)


################################################################################
# Subdirectories
################################################################################
pybind11_add_module(polysolve_python src/polysolve.cpp)
set_target_properties(polysolve_python PROPERTIES OUTPUT_NAME polysolve)
target_compile_features(polysolve_python PRIVATE cxx_std_17)
target_link_libraries(polysolve_python PRIVATE pybind11::module pybind11::json polysolve::polysolve)
install(TARGETS polysolve_python LIBRARY DESTINATION .)
