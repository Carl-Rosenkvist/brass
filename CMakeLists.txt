cmake_minimum_required(VERSION 3.14)
project(brass)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type specified. Forcing Release.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# ---- Force LTO/IPO ----
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_msg)
if(ipo_supported)
  # Global IPO for optimized configs
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)
else()
  message(WARNING "IPO/LTO not supported by this toolchain: ${ipo_msg}")
endif()

include_directories(include)

# ---- Explicit source lists (no GLOBs) ----
set(SRC_FILES
    src/analysis.cc
    src/analysisregister.cc
    src/binaryreader.cc
    src/mergekey.cc
    src/quantities.cc 
    src/blocks.cc 
    src/accessor.cc 
)

set(ANALYSIS_FILES
    analyses/bulk.cc
    analyses/centrality.cc
)


option(BUILD_PYTHON "Build Python extension module" ON)

if(BUILD_PYTHON)
  set(PYBIND11_FINDPYTHON ON)
  find_package(pybind11 CONFIG REQUIRED)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

  pybind11_add_module(_brass
      pythonlib/bindings.cc
      ${SRC_FILES}
      ${ANALYSIS_FILES}
  )
  target_include_directories(_brass PRIVATE include)

  set_target_properties(_brass PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/pythonlib/brass"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/pythonlib/brass"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/pythonlib/brass"
    OUTPUT_NAME "_brass"
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
  )

  add_executable(unit_tests
      ${TEST_SOURCES}
      ${SRC_FILES}
      ${ANALYSIS_FILES}
  )
  target_include_directories(unit_tests PRIVATE tests include)

  add_test(NAME unit_tests COMMAND unit_tests)
endif()
