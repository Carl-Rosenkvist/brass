# Changelog

All notable changes to this project will be documented in this file.

The version number follows **X.Y.Z**, where:
- **X** — milestone achievements  
- **Y** — new features  
- **Z** — bug fixes or small updates  

---

## [0.1.0] – 2025-10-23  
**First public release**

### Input / Output
- Introduced the initial **binary reader** for particle data files.
- Implemented YAML configuration loading for each run.
- Added basic metadata extraction using dotted keys.
- Introduced the first stable on-disk output directory structure for analyses.

### Added
- Core **analysis execution pipeline** (C++ backend + Python glue).
- Initial **analysis registry system**, enabling discovery and execution of analyses.
- Support for **simple C++ and Python analyses**, including histogramming and metadata-based grouping.
- `brass-analyze` command-line tool:
  - run scanning using `--pattern`
  - analysis selection
  - YAML loading
  - metadata building via `MetaBuilder`
- Quantity consistency checking across runs (warn-only mode in 0.1.0).
- Initial configuration expansion (list/dict flattening) for metadata.

### Changed
- Initial public API definitions established (baseline for later versioning).

### Fixed
- (No bugfixes recorded — first release)

### Removed
- (Not applicable — first release)

## [0.2.0] – 2025-11-??  
Second public release: unified merging and multiprocessing.

### Input / Output
- Added **strict binary filename matching** for `brass-analyze`:  
  the CLI now aborts with an error if no binary file in a run directory matches
  the patterns passed via `--binary-names`.
- Introduced **Python dict–based result merging** for analyses, providing a unified
  way to combine results from C++- and Python-based analyses.
- Added support for saving merged analysis results to **pickle** files, using the
  same dict-based representation.

### Added
- **Multiprocessing** support for analyses via `--nproc`, allowing parallel
  execution over multiple input files.
- Verbose CLI diagnostics for **discovered binary files**: in `--verbose` mode,
  `brass-analyze` now prints which binary file is used for each run directory.
- Python analysis lifecycle hooks to control merging:
  - `py::dict finalize(py::dict results) override`:
    ```cpp
    py::dict finalize(py::dict results) override {
        py::gil_scoped_acquire gil;
        if (py::hasattr(obj_, "finalize")) obj_.attr("finalize")(results);
        return results;
    }
    ```
    This lets Python analyses optionally implement a `finalize(results)` method
    to post-process or normalize their merged results.
  - `py::dict to_state_dict() const override`:
    ```cpp
    py::dict to_state_dict() const override {
        py::gil_scoped_acquire gil;
        if (!py::hasattr(obj_, "to_state_dict"))
            throw std::runtime_error("PythonAnalysis: missing to_state_dict()");
        return obj_.attr("to_state_dict")().cast<py::dict>();
    }
    ```
    Python analyses are now required to implement `to_state_dict()`, which
    defines the serializable state used for merging and persistence.

### Changed
- Unified the result handling for C++ and Python analyses around a shared
  **dict-based state and merging** model.
- Tightened the CLI behavior around binary file selection by removing the
  automatic `*.bin` fallback; users must now be explicit and consistent
  with `--binary-names`.

### Fixed
- (No specific bug fixes recorded for this release.)

### Removed
- Legacy fallback logic that would silently pick the first `*.bin` file in a
  run directory when no explicit `--binary-names` match was found.
