# SHAP approximation accuracy & runtime

For each dataset in `config/shap_approximation.yaml`, compares PDD-SHAP against
`shap`'s Sampling/Permutation/Kernel explainers on the same background and test
samples: accuracy (R2, Spearman rank correlation of PDD-SHAP vs. each
baseline) and runtime (surrogate training time, explanation time per method).

```bash
uv run python scripts/shap_approximation/run_experiment.py
uv run python scripts/shap_approximation/plot.py
```

Pass `-c/--config` to either script to use an alternative experiment config
YAML instead of the default `config/shap_approximation.yaml`.

Each run of `run_experiment.py` writes to a fresh
`OUT_DIR/YYYYMMDDhhmmss_shap_approximation/` directory (`OUT_DIR` from
`config/general.yaml`) — previous runs are never overwritten:
- `config.yaml` — the fully resolved configuration used for this run. Pass it
  back in as `--config` to reproduce the run exactly.
- `<dataset>/X_bg.csv`, `<dataset>/X_test.csv` — the background/test samples used.
- `<dataset>/pdd_shap.h5`, `<dataset>/<method>_shap.h5` — raw explanations per method.
- `metrics.csv` — one row per (dataset, baseline method): R2 and Spearman
  rank correlation of PDD-SHAP against that baseline.
- `runtimes.csv` — one row per (dataset, method): training time (PDD-SHAP
  only) and explanation time.

`plot.py` reads the most recent such run directory by default (`plots/*.png`
are written inside it); pass `-r/--run-dir` to plot a specific one.
