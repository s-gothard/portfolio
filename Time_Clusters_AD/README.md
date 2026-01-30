# Trajectory Clusters Benchmark

A minimal, runnable benchmarking suite for trajectory clustering methods on longitudinal clinical data. This is an **outline/starting point** implementation with mock models and synthetic data generation.

## What This Is

A thin framework for benchmarking unsupervised trajectory clustering approaches on cognitive decline data:

- **Mock implementations** of HMM (using KMeans on subject means) and DTW k-means (using slope/intercept clustering)
- **Synthetic data generation** for testing and development
- **Preprocessing pipeline** with time grid alignment and feature scaling
- **Evaluation framework** with internal clustering metrics
- **CLI interface** with `uv` for reproducible environments

## 60-Second Quickstart

```bash
# 1. Set up environment with uv
uv venv && uv sync

# 2. Generate synthetic data
uv run tb make-dataset --synthetic true

# 3. Preprocess data (align to annual grid, impute, scale)
uv run tb preprocess

# 4. Fit mock HMM model
uv run tb fit --method hmm --k 3 --data data/processed/dataset.parquet --out runs/hmm_mock.pkl

# 5. Evaluate clustering quality
uv run tb evaluate --method hmm --data data/processed/dataset.parquet --model runs/hmm_mock.pkl --out reports/metrics.json

# 6. Generate analysis report
uv run tb report --run runs/hmm_mock.pkl --out reports/outline_report.md --data data/processed/dataset.parquet
```

## Requirements

- **Python 3.13+** (supports 3.13t as well)
- **uv** for dependency management and virtual environments
- Core dependencies: numpy, pandas, scikit-learn, typer, pydantic, matplotlib

## CLI Commands

All functionality is accessed through the `tb` command:

### Data Generation
```bash
# Generate synthetic longitudinal data
tb make-dataset --synthetic true --n-subjects 200 --n-clusters 3

# Validate external data file (create configs/data.yaml first to map columns)
tb make-dataset --synthetic false --out path/to/your/data.csv --config configs/data.yaml
```

### Preprocessing
```bash
# Preprocess with time grid alignment and feature scaling
tb preprocess --in data/raw/synth.parquet --out data/processed/dataset.parquet

# Custom preprocessing config
tb preprocess --config configs/preprocess.yaml --data-config configs/data.yaml
```

### Model Fitting
```bash
# Fit mock HMM (currently KMeans on subject means)
tb fit --method hmm --k 3 --data data/processed/dataset.parquet

# Fit mock DTW k-means (currently slope/intercept clustering)
tb fit --method dtw_kmeans --k 4 --random-state 42
```

### Evaluation
```bash
# Evaluate with internal metrics (silhouette, Davies-Bouldin, etc.)
tb evaluate --method hmm --model runs/hmm_mock.pkl --out reports/metrics.json
```

### Reporting
```bash
# Generate markdown report with cluster summaries
tb report --run runs/hmm_mock.pkl --out reports/analysis.md
```

## Configuration

The repository works **out-of-the-box** with sensible defaults for synthetic data. For real UDS3/clinical data, you can optionally create config files:

- **`configs/data.yaml`**: Column mappings for your UDS3/clinical data exports
- **`configs/preprocess.yaml`**: Time grid alignment, imputation, scaling options  
- **`configs/methods.yaml`**: Model hyperparameters for HMM, DTW k-means, baselines
- **`configs/eval.yaml`**: Evaluation metrics and external validation toggles

Use `--config` flags to specify custom configurations, or omit them to use defaults.

## Repository Structure

```
.
├── README.md
├── LICENSE
├── pyproject.toml          # PEP 621 + hatchling; uv-compatible
├── uv.lock                 # Dependency lock file
├── .pre-commit-config.yaml
├── .gitignore
├── data/                   # Gitignored data directories
│   ├── raw/.gitkeep
│   └── processed/.gitkeep
└── src/trajectory_benchmark/
    ├── __init__.py
    ├── cli.py              # Typer CLI (main interface)
    ├── io/                 # Data loading with Pydantic validation
    ├── preprocess/         # Time grid alignment, feature preprocessing
    ├── models/             # Mock HMM and DTW k-means implementations
    ├── eval/               # Internal metrics, external validation stubs
    ├── viz/                # Trajectory and matrix plotting utilities
    └── utils/              # Synthetic data generator
```

## Roadmap / TODO

This is an **outline implementation**. Key items to replace/extend:

### 🔄 Replace Mock Models
- [ ] **Mock HMM** → pomegranate Gaussian-HMM with proper temporal modeling
- [ ] **Mock DTW k-means** → tslearn TimeSeriesKMeans(metric="dtw") with barycenters

### 🧬 Add Advanced Methods
- [ ] Latent Transition Analysis (LTA-style discrete state models)
- [ ] Growth Mixture Models
- [ ] Spectral clustering on trajectory embeddings
- [ ] Hierarchical clustering with DTW distances

### 📊 Expand Evaluation
- [ ] **Transition matrices** and n-step risk estimation
- [ ] **External validation**: diagnosis alignment (ARI/NMI vs CN/MCI/dementia)
- [ ] **Enrichment testing**: APOE ε4 and amyloid positivity by cluster
- [ ] **Neuropathology association**: logistic regression with covariates

### 🎨 Enhance Visualization
- [ ] DTW barycenter trajectory plots
- [ ] Sankey diagrams for state transitions over time
- [ ] Interactive plotly visualizations
- [ ] State characterization heatmaps with statistical annotations

### 🔧 Production Features
- [ ] MLflow experiment tracking
- [ ] Cross-validation and model selection
- [ ] Bootstrap confidence intervals for metrics
- [ ] HTML report generation with embedded plots
- [ ] Docker containerization

## Data Privacy

- All `data/` directories are gitignored
- No real clinical data is committed to the repository
- Synthetic data generator creates realistic-looking test data only
- Configure your real data column mappings in `configs/data.yaml`

## Development

```bash
# Install development dependencies
uv sync --group dev

# Set up pre-commit hooks
pre-commit install

# Run linting and formatting
ruff check src/
ruff format src/

# Generate new synthetic data for testing
uv run tb make-dataset --synthetic true --n-subjects 100 --n-clusters 2
```

## License

Apache 2.0 - See [LICENSE](LICENSE) file.

---

**Note**: This repo does *not* estimate population prevalence (non-population sample). It focuses on trajectory clustering methodology and validation against clinical outcomes.