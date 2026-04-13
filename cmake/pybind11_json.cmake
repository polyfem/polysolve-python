# pybind11_json
# License: MIT

if(TARGET pybind11::json)
    return()
endif()

message(STATUS "Third-party: creating target 'pybind11::json'")

set(BUILD_TESTS		OFF CACHE BOOL "" FORCE)
set(DOWNLOAD_GTEST	OFF CACHE BOOL "" FORCE)
set(PYBIND11_FINDPYTHON ON)

include(CPM)
CPMAddPackage(
    pybind11_json
    GIT_REPOSITORY https://github.com/pybind/pybind11_json.git
    GIT_TAG 0.2.15
    GIT_SHALLOW FALSE
)


add_library(pybind11::json ALIAS pybind11_json)