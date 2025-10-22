#!/usr/bin/env bash
set -euo pipefail

# Absolute or relative paths to top-level C++ source directories
CPP_DIRS="../src ../include ../pythonlib"

echo "🔹 Formatting C++ (.h / .cc) files with clang-format..."
# Use `-print0` and `--no-run-if-empty` to avoid errors if no files match
find $CPP_DIRS -type f \( -iname '*.h' -o -iname '*.cc' \) -print0 \
    | xargs -0 --no-run-if-empty clang-format -i

echo "🔸 Formatting Python files with ruff..."
ruff format ../pythonlib

echo "✅ All formatting done."
