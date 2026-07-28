import argparse
import csv
import glob
import os
import time
import warnings

import joblib
import numpy as np
from shap import KernelExplainer, PermutationExplainer, SamplingExplainer
from tqdm import tqdm

from experiments.download_datasets import get_dataset, get_pred_type, get_valid_datasets

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


def compute_shapley_values(pred_fn, X_bg, X, explainer: str):
    if explainer == "permutation":
        explainer = PermutationExplainer(pred_fn, X_bg)
        values = explainer(X).values
        return values if len(values.shape) == 3 else np.expand_dims(values, -1)
    elif explainer == "kernel":
        explainer = KernelExplainer(pred_fn, X_bg)
        values = explainer.shap_values(X)
        if type(values) == list:
            return np.stack(values, axis=-1)
        else:
            return np.expand_dims(values, -1)
    elif explainer == "sampling":
        explainer = SamplingExplainer(pred_fn, X_bg)
        values = explainer(X).values
        return values if len(values.shape) == 3 else np.expand_dims(values, -1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=_DESC, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=str,
        default="./data",
        help="Directory where datasets and models can be found, and where the background distributions"
        "and Shapley values should be stored. This corresponds to OUT_DIR of train_models.py.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        choices=get_valid_datasets(),
        nargs="*",
        help="The dataset(s) to use. By default, all datasets in OUT_DIR are used.",
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
    datasets = args.datasets if args.datasets is not None else os.listdir(args.out_dir)
    for ds in datasets:
        if ds not in os.listdir(args.out_dir):
            raise ValueError(f"Dataset {ds} not found in {args.out_dir}.")

    prog = tqdm(datasets)
    for ds_name in prog:
        # Load data
        ds_dir = os.path.join(args.out_dir, ds_name)
        ds_shap_dir = os.path.join(ds_dir, "shap")
        os.makedirs(ds_shap_dir, exist_ok=True)
        prog.set_postfix({"dataset": ds_name})
        prog.set_description("Loading data...")
        X_train, X_test, y_train, y_test = get_dataset(
            ds_name, args.out_dir, download=False
        )
        pred_type = get_pred_type(ds_name)

        # Extract background set and save to disk
        num_bg = min(args.num_bg, X_train.shape[0])
        if num_bg < args.num_bg:
            warnings.warn(
                f"{ds_name} train set only contains {num_bg} rows."
                f"Using full train set as background set."
            )
        X_bg = X_train.sample(n=num_bg)
        X_bg.to_csv(os.path.join(ds_shap_dir, "X_bg.csv"), index=False)

        # Get test samples and save to disk
        X_test_shap = X_test
        if args.num_test > 0:
            num_test = min(args.num_test, X_test.shape[0])
            if num_test < args.num_test:
                warnings.warn(
                    f"{ds_name} test set only contains {num_test} rows. Using full test set."
                )
            X_test_shap = X_test.sample(n=num_test)
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
            pred_fn = (
                model.predict_proba if pred_type == "classification" else model.predict
            )

            # Compute Shapley values using shap explainers
            model_dir = os.path.join(ds_shap_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            with open(os.path.join(model_dir, "runtimes.csv"), "w") as fp:
                writer = csv.DictWriter(fp, ["explainer", "runtime (s)"])
                writer.writeheader()
                for explainer in explainers:
                    start_t = time.time()
                    values = compute_shapley_values(
                        pred_fn, X_bg.to_numpy(), X_test_shap.to_numpy(), explainer
                    )
                    end_t = time.time()
                    writer.writerow(
                        {"explainer": explainer, "runtime (s)": end_t - start_t}
                    )
                    # We save to .npy instead of .csv because the output is 3-dimensional (row, column, output)
                    np.save(os.path.join(model_dir, f"{explainer}.npy"), values)
