# Changelog

All notable changes to this project will be documented in this file.

The version number follows **X.Y.Z**, where:
- **X** — milestone achievements  
- **Y** — new features  
- **Z** — bug fixes or small updates  


## [0.3.0] – Unreleased  
Incremental improvements focused on usability and analysis utilities.

### Added
- General **decay reconstruction utility** (`DecayReconstructor`) for building resonances from daughter particles.
- Initial unit tests for reconstruction logic.
- New CLI tool **`brass-scan`** for generating SMASH command-line scans directly from the terminal.
- Support for intuitive parameter input via `--param KEY=val1,val2,...`.

### Improved
- Simplified **scan interface** in `brass.scan`:
  - Removed split-specific logic in favor of a unified job model.
  - Clear separation between *parameter scan* and *job splitting* (`events_per_job`, `max_events`).
- Cleaner command generation via `sweep_cmds(prefix=...)`.
- Minor internal cleanups in analysis and merge handling.

### Changed
- Scan workflow now defines **jobs per parameter point** using:
  - `events_per_job`
  - `max_events`
- Removed direct dependency of `MergeKey` on YAML serialization (cleaner separation of concerns).

### Fixed
- Various small fixes in CLI handling and module imports.

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


