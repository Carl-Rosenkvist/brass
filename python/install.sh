#!/bin/bash
set -e


PYBIND11_DIR=$(python -m pybind11 --cmakedir)
echo "Using pybind11 CMake config from: $PYBIND11_DIR"

pip install -e . \
  --config-settings=cmake.define.pybind11_DIR="$PYBIND11_DIR" \
  --config-settings=cmake.define.WITH_PYBIND=ON

echo "✅ Installation complete"

