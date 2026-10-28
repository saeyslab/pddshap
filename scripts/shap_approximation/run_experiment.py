import argparse
import csv
import datetime
import pathlib

import numpy as np
import yaml
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

from pddshap import PartialDependenceDecomposition
from pddshap.evaluation import compare_values, time_call
from pddshap.experiments import (
    compute_shapley_values,
    load_experiment_config,
    load_dataset,
    make_model,
    predict_fn,
    sample_rows,
    save_shap_values,
)
from pddshap.sampling import IndependentConditioningMethod, RandomSubsampleCollocation

_DESC = """
For each dataset configured in config/shap_approximation.yaml, this script:
  1. Trains a gradient boosting model and fits a PartialDependenceDecomposition
     (PDD) on a background sample drawn from the training set.
  2. Uses the fitted PDD to generate PDD-SHAP explanations for a sample of
     test instances.
  3. Runs the configured `shap` baseline explainers (sampling, permutation,
     kernel) on the same background/test samples.
  4. Compares PDD-SHAP against each baseline (R2, Spearman rank correlation)
     and records runtimes for every method.

Each run writes to a fresh OUT_DIR/YYYYMMDDhhmmss_shap_approximation/ directory
(never overwriting a previous run's results): per-dataset HDF5 files with the
raw explanations, two experiment-wide CSVs (metrics.csv, runtimes.csv) used by
plot.py, and a config.yaml snapshot of the resolved configuration used for
this run. Passing that config.yaml back in via --config reproduces the run.
"""

_ESTIMATOR_REGISTRY = {
    "knn": KNeighborsRegressor,
    "tree": DecisionTreeRegressor,
    "forest": GradientBoostingRegressor,
}


def _append_csv_row(path: pathlib.Path, fieldnames: list[str], row: dict) -> None:
    write_header = not path.is_file()
    with open(path, "a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_dataset(
    ds_name: str,
    data_id: int,
    model_kwargs: dict,
    pdd_cfg: dict,
    baseline_methods: list[str],
    n_background: int,
    n_test: int,
    random_seed: int,
    data_dir: str,
    exp_out_dir: pathlib.Path,
) -> None:
    """Run the full PDD-SHAP-vs-baselines comparison for a single dataset."""
    # Seeds every source of randomness not otherwise pinned by an explicit
    # random_state/seed argument (e.g. RandomSubsampleCollocation's subsampling,
    # shap's baseline explainers), so a run is reproducible from `random_seed` alone.
    np.random.seed(random_seed)

    split = load_dataset(data_dir, openml_data_id=data_id)

    ds_out_dir = exp_out_dir / ds_name
    ds_out_dir.mkdir(parents=True, exist_ok=True)

    X_bg = sample_rows(split.X_train, n_background, random_state=random_seed, label=f"{ds_name} background")
    X_test = sample_rows(split.X_test, n_test, random_state=random_seed, label=f"{ds_name} test")
    X_bg.to_csv(ds_out_dir / "X_bg.csv", index=False)
    X_test.to_csv(ds_out_dir / "X_test.csv", index=False)

    print(f"[{ds_name}] Training gradient boosting ({split.pred_type}) model...")
    model = make_model(split.pred_type, "gradientboosting", random_state=random_seed, **model_kwargs)
    model.fit(split.X_train.to_numpy(), split.y_train)
    pred_fn = predict_fn(model, split.pred_type)

    print(f"[{ds_name}] Fitting PDD...")

    def _fit_pdd():
        pdd = PartialDependenceDecomposition(
            model=pred_fn,
            collocation_method=RandomSubsampleCollocation(frac=pdd_cfg["collocation_frac"]),
            conditioning_method=IndependentConditioningMethod(X_bg.to_numpy()),
            estimator_type=_ESTIMATOR_REGISTRY[pdd_cfg["estimator"]],
            estimator_kwargs=pdd_cfg.get("estimator_kwargs", {}),
        )
        pdd.fit(X_bg, max_size=pdd_cfg["max_size"], n_jobs=pdd_cfg.get("n_jobs", 1))
        return pdd

    pdd, pdd_train_time = time_call(_fit_pdd)

    print(f"[{ds_name}] Computing PDD-SHAP explanations...")
    pdd_shap, pdd_explain_time = time_call(pdd.shapley_values, X_test)
    save_shap_values(pdd_shap, list(X_test.columns), str(ds_out_dir / "pdd_shap.h5"))

    runtimes_path = exp_out_dir / "runtimes.csv"
    runtime_fields = ["dataset", "method", "train_time (s)", "explain_time (s)"]
    _append_csv_row(
        runtimes_path, runtime_fields,
        {"dataset": ds_name, "method": "pdd_shap", "train_time (s)": pdd_train_time, "explain_time (s)": pdd_explain_time},
    )

    metrics_path = exp_out_dir / "metrics.csv"
    metrics_fields = ["dataset", "method", "r2", "spearman"]
    for method in baseline_methods:
        print(f"[{ds_name}] Computing '{method}' baseline explanations...")
        values, elapsed = time_call(
            compute_shapley_values, pred_fn, X_bg.to_numpy(), X_test.to_numpy(), method
        )
        save_shap_values(values, list(X_test.columns), str(ds_out_dir / f"{method}_shap.h5"))
        _append_csv_row(
            runtimes_path, runtime_fields,
            {"dataset": ds_name, "method": method, "train_time (s)": "", "explain_time (s)": elapsed},
        )

        stats = compare_values(values, pdd_shap)
        _append_csv_row(
            metrics_path, metrics_fields,
            {"dataset": ds_name, "method": method, "r2": stats["r2"], "spearman": stats["spearman"]},
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=_DESC, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "-c", "--config", type=pathlib.Path, default=None,
        help="Path to an alternative experiment config YAML, overriding the "
             "default config/shap_approximation.yaml.",
    )
    args = parser.parse_args()

    cfg = load_experiment_config(pathlib.Path(__file__).parent, config_path=args.config)
    pdd_cfg = cfg["pdd"]
    random_seed = int(cfg.get("random_seed", 42))

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    exp_out_dir = pathlib.Path(cfg["out_dir"]) / f"{timestamp}_shap_approximation"
    exp_out_dir.mkdir(parents=True)

    with open(exp_out_dir / "config.yaml", "w") as fp:
        yaml.safe_dump(cfg, fp, default_flow_style=False, sort_keys=False)

    for ds_name, ds_info in cfg["datasets"].items():
        print(f"=== {ds_name} ===")
        run_dataset(
            ds_name=ds_name,
            data_id=ds_info["data_id"],
            model_kwargs={k: v for k, v in ds_info.items() if k not in ("data_id", "pred_type")},
            pdd_cfg=pdd_cfg,
            baseline_methods=cfg["baseline_methods"],
            n_background=int(cfg["n_background"]),
            n_test=int(cfg["n_test"]),
            random_seed=random_seed,
            data_dir=cfg["data_dir"],
            exp_out_dir=exp_out_dir,
        )

    print("Done.")
