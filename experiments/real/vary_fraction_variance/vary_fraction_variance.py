import argparse
import csv

import joblib
import glob
from tqdm import tqdm
import time
from experiments.real.datasets import get_pred_type, get_dataset
import os
from pddshap import PartialDependenceDecomposition, RandomSubsampleGenerator
import pandas as pd
import numpy as np


def _convert_dtypes(df):
    int_cols = {c: np.int8 for c in df.select_dtypes(include="int64").columns}
    float_cols = {c: np.float32 for c in df.select_dtypes(include="float64").columns}
    return df.astype({**int_cols, **float_cols})


_DESC = """
This script computes Shapley values using PDD-SHAP for varying fractions of variance explained.
It requires a data directory containing the datasets, models and original shapley values.
This corresponds to the OUT_DIR from compute_shap.py.

Results are saved in a separate directory OUT_DIR.
For each fraction of variance x.yy and model, the results are saved in OUT_DIR/DATASET_NAME/MODEL_NAME/xyy.
For example: for dataset adult, fraction of variance 0.95 and model knn, results are saved in OUT_DIR/adult/knn/095.

For each dataset, runtimes are saved in OUT_DIR/DATASET_NAME/runtimes.csv.
This file contains the runtime for training and inference separately, for each model and fraction of variance.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=_DESC)
    parser.add_argument("-d", "--data-dir", type=str, default="./data",
                        help="Directory where the output of compute_shap.py was saved.")
    parser.add_argument("-o", "--out-dir", type=str, default="./out",
                        help="Output directory where results should be stored.")
    parser.add_argument("-e", "--variance-explained", type=float, nargs="*", default=[0.9],
                        help="The different fractions of variance explained. Default: 0.9."
                             "This argument can take multiple values.")
    parser.add_argument("--datasets", type=str, nargs="*",
                        help="Datasets to use. By default, all datasets in DATA_DIR will be used.")
    parser.add_argument("--estimator", type=str, choices=["tree", "forest", "knn"], default="tree",
                        help="Model to use to estimate PDD components")
    parser.add_argument("--models", type=str, nargs="*",
                        help="Models to use. By default, all available models will be used.")
    parser.add_argument("-k", type=int, default=None,
                        help="Parameter k for KNN estimator. Ignored if estimator is not knn.")
    args = parser.parse_args()

    # Check arguments for validity and default values
    os.makedirs(args.out_dir, exist_ok=True)
    datasets = args.datasets if args.datasets is not None else os.listdir(args.data_dir)
    for ds in datasets:
        if ds not in os.listdir(args.data_dir):
            raise ValueError(f"Dataset {ds} not found in {args.data_dir}.")
    for var in args.variance_explained:
        if not 0 < var <= 1:
            raise ValueError(f"Illegal value for variance-explained: {var}")

    prog = tqdm(datasets)
    for ds_name in prog:
        ds_out_dir = os.path.join(args.out_dir, ds_name)
        os.makedirs(ds_out_dir, exist_ok=True)
        ds_data_dir = os.path.join(args.data_dir, ds_name)
        ds_model_dir = os.path.join(ds_data_dir, "models")
        ds_shap_dir = os.path.join(ds_data_dir, "shap")
        pred_type = get_pred_type(ds_name)

        model_names = args.models
        if model_names is None:
            model_names = [os.path.basename(fn)[:-4] for fn in glob.glob(os.path.join(ds_model_dir, "*.pkl"))]
        with open(os.path.join(ds_out_dir, "runtimes.csv"), "w") as runtime_fp:
            writer = csv.DictWriter(runtime_fp, ["model", "fraction", "training", "inference"])
            writer.writeheader()
            for model_name in model_names:
                # Load model
                model = joblib.load(os.path.join(ds_model_dir, f"{model_name}.pkl"))
                pred_fn = model.predict_proba if pred_type == "classification" else model.predict

                for variance_explained in args.variance_explained:
                    print(variance_explained)
                    # Create output subdirectory
                    dirname = '1' if variance_explained == 1.0 else str(variance_explained).replace('.', '')
                    print(dirname)
                    var_out_dir = os.path.join(ds_out_dir, model_name, dirname)
                    os.makedirs(var_out_dir, exist_ok=True)

                    # Load datasets
                    X_train, _, _, _ = get_dataset(ds_name, args.data_dir, download=False)
                    X_bg = _convert_dtypes(pd.read_csv(os.path.join(ds_shap_dir, "X_bg.csv")))
                    X_test = _convert_dtypes(pd.read_csv(os.path.join(ds_shap_dir, "X_test.csv")))

                    # Train PDD-SHAP model
                    est_kwargs = {}
                    if args.k is not None:
                        est_kwargs["k"] = args.k
                    decomposition = PartialDependenceDecomposition(pred_fn, RandomSubsampleGenerator(),
                                                                   args.estimator, est_kwargs)
                    start_t = time.time()
                    decomposition.fit(X_bg, X_train, variance_explained=variance_explained)
                    end_t = time.time()
                    fit_time = end_t - start_t

                    # Estimate Shapley values on the test set
                    start_t = time.time()
                    values = decomposition.shapley_values(X_test)
                    end_t = time.time()
                    infer_time = end_t - start_t

                    # Get surrogate model output on test set
                    output = decomposition(X_test)

                    # Save results to disk and write timings to csv
                    writer.writerow({"model": model_name, "fraction": variance_explained,
                                     "training": fit_time, "inference": infer_time})
                    np.save(os.path.join(var_out_dir, "values.npy"), values)
                    np.save(os.path.join(var_out_dir, "output.npy"), output)
