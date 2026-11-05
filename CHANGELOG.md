# Changelog

All notable changes to this project will be documented in this file.

The version number follows **X.Y.Z**, where:
- **X** — milestone achievements  
- **Y** — new features  
- **Z** — bug fixes or small updates  

## [0.4.0] – 2026-11-05
Rewrite of the core: a block-based binary reader, a Boost.Histogram-based
histogram engine, and analyses written in Python. **This release is not
backwards compatible with 0.3.x** (see *Changed* and *Removed*).

### Added
- **Block-based reader**: `BinaryReader.read()` returns `ParticleBlock`,
  `InteractionBlock` (incoming/outgoing particles, `rho`, `sigma`, `sigma_p`,
  `process`) and `EndBlock` (`impact_parameter`, `empty`), with optional
  skipping of elastic two-particle events.
- **Zero-copy NumPy columns** of particle quantities via `Particles.column()`
  and `Particles.columns()`.
- **Histogram engine** built on Boost.Histogram (`HistogramRequest`,
  `HistogramRunner`, `brass.histogram()`, `brass.histograms()`):
  - 1D to 4D histograms with regular, variable and integer axes
  - several histograms filled in one pass over a file
  - grouped histograms, e.g. one per PDG code (`HistogramGroupBy`)
  - per-particle or per-interaction weights
  - histograms of particles, incoming/outgoing interaction particles,
    interaction blocks and end blocks (`HistogramTarget`)
  - computed quantities `pt`, `p`, `mt`, `y_rap` and `mass`
- `DileptonHistogrammer` for weighted dilepton invariant-mass spectra per
  decay channel.
- **Python analysis framework**: subclass `brass.Analysis`; `AnalysisRunner`
  runs analyses over many run folders in parallel and merges results with
  identical SMASH configuration (ignoring seeds, logging and output settings).
- New **`brass` CLI** with one subcommand per analysis; `--load FILE` adds
  analyses from your own Python files.
- Built-in analyses: `bulk` (y–mT histograms per PDG, with `--add-pdgs` to
  combine species), `centrality` (impact parameter and charged multiplicity),
  `dileptons`.
- **Bulk emulation** (`brass.emulation`): PCA + Gaussian-process emulator for
  bulk observables and a χ² likelihood against experimental data. Requires the
  new `emulation` extra (`pip install "pybrass[emulation]"`).
- **Experimental data** collection in `exp_data/` (dN/dy for Pb+Pb, Au+Au and
  pp), provided as-is.
- **`brass.testing`**: write synthetic SMASH binary files and run folders, for
  tests, benchmarks and examples.
- **Prebuilt wheels** for Python 3.11–3.14 on Linux (x86_64) and macOS
  (Apple Silicon, macOS 11+), built and tested automatically for each release.
  Previously only a macOS wheel for Python 3.12 was published.
- Container build (`container/Dockerfile`).
- Benchmark script with a NumPy comparison, and a *Performance* section in the
  README.
- `brass-scan` accepts list values, e.g. `Modi.Collider.Impact.Range=[0,3],[3,6]`.

### Improved
- Particle quantities are resolved from names once; reading, computing and
  filling no longer compare strings per particle.
- Histograms are computed and filled column-wise per block; BRASS is faster
  than an equivalent NumPy reader for both small and large events (see
  *Performance* in the README).
- Particle quantity types (int32/double) are explicit instead of inferred from
  their size.

### Changed
- **Requires Python 3.11 or newer**; wheels are built for Python 3.11–3.14.
- Analyses are now Python classes instead of C++ analyses registered in the
  extension module.
- `brass-scan` and `brass-merge` were rewritten; `brass-merge` merges result
  pickles by their SMASH configuration.
- `DecayReconstructor` moved to `brass.decays` (still available as
  `brass.DecayReconstructor`).
- Optional extras are now `emulation` and `test`; the `dev` extra was removed.
- `mt` and `y_rap` are read from the file when stored, and computed otherwise.
- Unknown quantity names now raise when a histogram is set up, instead of
  while reading.
- The README badge and `CITATION.cff` use the Zenodo DOI for all versions
  (`10.5281/zenodo.17426761`), which always resolves to the latest release.

### Removed
- CLI tools `brass-analyze`, `brass-inspect` and `brass-run_cmds`
  (use `brass` and `brass-merge`).
- C++ analysis registry (`Analysis`, `MergeKey`, accessors, 1D/2D histogram
  classes) and the old analyses `dndydmt`, `dndydmt_unwounded`,
  `count_unwounded` and `xsection`.
- `HistND`, `MetaBuilder`, `smash_cmd` and the `brass.merging` module.

### Fixed
- `DataHandler.from_file` always raised a `TypeError`.
- Merging two `dileptons` results raised an `AttributeError`.

## [0.3.1] – 2026-05-01
Small cleanups and addition of the merge CLI tool.

### Added
- Exposed the existing merge CLI tool in the Python library.
- Empty data field in ``EndBlock``is accessable now. 

### Improved
- Simplified `DecayReconstructor`, reducing unnecessary complexity and improving readability and runtime.

### Changed
- (None)

### Fixed
- Integrated previously unused merge CLI tool into the Python interface.

## [0.3.0] – 2026-04-25  
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


