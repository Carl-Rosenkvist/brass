cmake_minimum_required(VERSION 3.14)
project(brass)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

set(SRC_FILES
    src/binaryreader.cc
    src/histogram.cc
    src/histogram_runner.cc
)
set(BOOST_HEADER_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/histogram/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/assert/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/config/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/core/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/mp11/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/throw_exception/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/variant2/include
)

option(BUILD_PYTHON "Build Python extension module" ON)

if(BUILD_PYTHON)
  set(PYBIND11_FINDPYTHON ON)

  find_package(pybind11 CONFIG REQUIRED)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

  pybind11_add_module(_brass
      src/bindings.cc
      ${SRC_FILES}
  )

  target_include_directories(_brass PRIVATE
      include
      ${BOOST_HEADER_DIRS}
  )

  install(TARGETS _brass DESTINATION brass)
endif()

# ---- Tests ----
option(BUILD_TESTS "Build unit tests" OFF)

if(BUILD_TESTS)
  enable_testing()

  set(TEST_SOURCES
      tests/main.cc
      tests/binaryreader.cc
      tests/histogram.cc
  )

  add_executable(unit_tests
      ${TEST_SOURCES}
      ${SRC_FILES}
  )

  target_include_directories(unit_tests PRIVATE
      tests
      include
      ${BOOST_HEADER_DIRS}
  )

  add_test(NAME unit_tests COMMAND unit_tests)
endif()
