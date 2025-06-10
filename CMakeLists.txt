cmake_minimum_required(VERSION 3.14)
project(brass)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Include headers
include_directories(include)

# ---- Explicit source lists (no GLOBs) ----
set(SRC_FILES
    src/analysis.cc
    src/analysisregister.cc
    src/binaryreader.cc
    src/datatree.cc
    src/mergekey.cc
    src/main.cc
)

set(ANALYSIS_FILES
    analyses/bulkd.cc
)

# Third-party: yaml-cpp (vendored)
add_subdirectory(external/yaml-cpp)

# ---- Executable ----
add_executable(brass
    ${SRC_FILES}
    ${ANALYSIS_FILES}
)
target_link_libraries(brass PRIVATE yaml-cpp)


# ------------- Python module ----------
option(BUILD_PYTHON "Build Python extension module" ON)

if(BUILD_PYTHON)
  set(PYBIND11_FINDPYTHON ON)
  find_package(pybind11 CONFIG REQUIRED)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

  pybind11_add_module(brass_py
      python/bindings.cc
      src/analysis.cc
      src/analysisregister.cc
      src/binaryreader.cc
      src/datatree.cc
      src/mergekey.cc
      analyses/rapidity_spectra.cc
  )
  target_include_directories(brass_py PRIVATE include)
  target_link_libraries(brass_py PRIVATE yaml-cpp)
  set_target_properties(brass_py PROPERTIES OUTPUT_NAME "brass")

  if(DEFINED SKBUILD_PLATLIB_DIR)
    install(TARGETS brass_py LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR})
  endif()
endif()
# ---- Tests ----
option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
    enable_testing()

    # test sources (only these two)
    set(TEST_SOURCES
        tests/main.cc
        tests/binaryreader.cc
    )

    # app sources WITHOUT main.cc
    set(SRC_NO_MAIN
        src/analysis.cc
        src/analysisregister.cc
        src/binaryreader.cc
        src/datatree.cc
        src/mergekey.cc
    )

    add_executable(unit_tests
        ${TEST_SOURCES}
        ${SRC_NO_MAIN}
        ${ANALYSIS_FILES}
    )
    target_include_directories(unit_tests PRIVATE tests include)
    target_link_libraries(unit_tests PRIVATE yaml-cpp)

    add_test(NAME unit_tests COMMAND unit_tests)
endif()
