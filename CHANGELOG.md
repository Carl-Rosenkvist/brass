# Changelog

All notable changes to this project will be documented in this file.

The version number follows **X.Y.Z**, where:
- **X** — milestone achievements  
- **Y** — new features  
- **Z** — bug fixes or small updates  

---

## [0.1.0] – 2025-10-23  
**First public release**

### Added
- Initial **binary reader** for particle data files.
- Basic metadata extraction using dotted keys.
- First stable **on-disk output directory structure**.
- Core **analysis execution pipeline** (C++ backend + Python glue).
- Initial **analysis registry system**, enabling discovery and execution of analyses.
- Support for simple **C++ and Python analyses**, including histogramming and metadata-based grouping.
- `brass-analyze` command-line tool:
  - run scanning using `--pattern`
  - analysis selection
  - YAML loading
  - metadata building via `MetaBuilder`
- Quantity consistency checking across runs (warn-only mode in 0.1.0).
- Initial configuration expansion (list/dict flattening) for metadata.

### Changed
- Established initial public API definitions (baseline for future versioning).

### Fixed
- (No bugfixes recorded — first release)

### Removed
- (Not applicable — first release)

---

## [0.2.0] – 2025-12-02  
Second public release: unified merging and multiprocessing.

### Added
- **Multiprocessing** support for analyses via `--nproc`.
- Unified **dict-based result merging** for C++ and Python analyses.
- Support for saving merged results to **pickle** using the shared dict representation.
- Verbose diagnostics: in `--verbose` mode, `brass-analyze`
  now prints which binary file is selected in each run directory.
- Analysis hooks:
  - Optional `finalize(results)` for post-processing the merged result.
  - Required `to_state_dict()` implementation for serialization and merging.
- New experimental command-line tool for running SMASH: brass-run_cmds

### Changed
- **Strict binary filename matching** in `brass-analyze`:  
  the CLI now aborts with an error if no file matches the patterns given by `--binary-names`.
- Unified state and merging model for both Python and C++ analyses.
- Removed automatic fallback to `*.bin`; users must now specify consistent names via `--binary-names`.

### Fixed
- (No specific bug fixes recorded for this release.)

### Removed
- Legacy silent fallback that selected the first `*.bin` file when no explicit name matched.
