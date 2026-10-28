import argparse
import csv
import glob
import os
import pathlib

import joblib
import numpy as np
from tqdm import tqdm

from pddshap.datasets import load
from pddshap.evaluation import time_call
from pddshap.experiments import compute_shapley_values, load_experiment_config, predict_fn, sample_rows

_EXPLAINERS = ["permutation"]

_DESC = """
This script extracts a background distribution from the training set and computes Shapley values on the test set
using existing explainers (SamplingExplainer, PermutationExplainer) in the shap library.

OUT_DIR is expected to contain datasets and models produced by train_models.py, and should be the same directory
that was given as OUT_DIR to that script.

The background distribution is saved to [OUT_DIR]/[DATASET_NAME]/shap/X_bg.csv.
Shapley values (sampling.csv and permutation.csv) and runtimes (runtimes.csv)
are saved to [OUT_DIR]/[DATASET_NAME]/shap/[MODEL_NAME].
Test samples (on which Shapley values were computed, which is a subset of the original test set)
are saved to [OUT_DIR]/[DATASET_NAME]/shap/X_test.csv.
"""


if __name__ == "__main__":
    cfg = load_experiment_config(pathlib.Path(__file__).parent)
    datasets_cfg = cfg["datasets"]

    parser = argparse.ArgumentParser(
        description=_DESC, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=str,
        default=cfg["data_dir"],
        help="Directory where datasets and models can be found, and where the background distributions"
        "and Shapley values should be stored. This corresponds to OUT_DIR of train_models.py.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        choices=list(datasets_cfg),
        nargs="*",
        help="The dataset(s) to use. By default, all datasets already cached in OUT_DIR are used.",
    )
    parser.add_argument(
        "--explainers",
        nargs="*",
        choices=_EXPLAINERS,
        help="Explainer(s) to use for computing Shapley values."
        "By default, all explainers are used.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Model(s) to use for computing Shapley values."
        "By default, all available models are used.",
    )
    parser.add_argument(
        "--num-test",
        type=int,
        default=0,
        help="Maximum number of test samples to compute Shapley values on."
        "By default, the full test set is used.",
    )
    parser.add_argument(
        "--num-bg",
        type=int,
        default=100,
        help="Number of background distribution samples. Default: 100.",
    )
    args = parser.parse_args()

    # Check arguments for validity and default values
    explainers = args.explainers if args.explainers is not None else _EXPLAINERS
    datasets = args.datasets if args.datasets is not None else list(datasets_cfg.keys())

    prog = tqdm(datasets)
    for ds_name in prog:
        # Load data
        ds_dir = os.path.join(args.out_dir, ds_name)
        ds_shap_dir = os.path.join(ds_dir, "shap")
        os.makedirs(ds_shap_dir, exist_ok=True)
        prog.set_postfix({"dataset": ds_name})
        prog.set_description("Loading data...")
        split = load(ds_name, args.out_dir, pred_type=datasets_cfg[ds_name]["pred_type"])

        # Extract background set and save to disk
        X_bg = sample_rows(split.X_train, args.num_bg, random_state=42, label=f"{ds_name} background")
        X_bg.to_csv(os.path.join(ds_shap_dir, "X_bg.csv"), index=False)

        # Get test samples and save to disk
        X_test_shap = split.X_test
        if args.num_test > 0:
            X_test_shap = sample_rows(split.X_test, args.num_test, random_state=42, label=f"{ds_name} test")
        X_test_shap.to_csv(os.path.join(ds_shap_dir, "X_test.csv"), index=False)

        # For each model and each explainer, compute Shapley values and save to disk
        model_names = (
            args.models
            if args.models is not None
            else [
                os.path.basename(fn)[:-4]
                for fn in glob.glob(os.path.join(ds_dir, "models", "*.pkl"))
            ]
        )
        for model_name in model_names:
            prog.set_description(f"Computing values for {model_name}")
            # Load the model
            model = joblib.load(os.path.join(ds_dir, "models", f"{model_name}.pkl"))
            pred_fn = predict_fn(model, split.pred_type)

            # Compute Shapley values using shap explainers
            model_dir = os.path.join(ds_shap_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            with open(os.path.join(model_dir, "runtimes.csv"), "w") as fp:
                writer = csv.DictWriter(fp, ["explainer", "runtime (s)"])
                writer.writeheader()
                for explainer in explainers:
                    values, elapsed = time_call(
                        compute_shapley_values,
                        pred_fn, X_bg.to_numpy(), X_test_shap.to_numpy(), explainer,
                    )
                    writer.writerow({"explainer": explainer, "runtime (s)": elapsed})
                    # We save to .npy instead of .csv because the output is 3-dimensional (row, column, output)
                    np.save(os.path.join(model_dir, f"{explainer}.npy"), values)
