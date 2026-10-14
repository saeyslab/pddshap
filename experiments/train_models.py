import argparse
import csv
import os
import pathlib

import joblib
from sklearn.metrics import balanced_accuracy_score, r2_score
from tqdm import tqdm

from pddshap.datasets import load
from pddshap.experiments import load_experiment_config, make_model, model_names

_DESC = """
This script downloads datasets, splits them into train and test set,
and trains and saves one or multiple models on them.

Each dataset is saved to ./data/[DATASET_NAME]. This folder contains the full dataset
(data.csv, labels.csv), as well as a pre-made train test split (X_train.csv, y_train.csv, X_test.csv, y_test.csv).

The models are saved to ./data/[DATASET_NAME]/models. Each model is pickled to [MODEL_NAME].pkl.
The models directory also contains a scores.csv file, which contains the balanced accuracy (in case of classification)
or R^2 (in case of regression) for each model.
"""


if __name__ == "__main__":
    cfg = load_experiment_config(pathlib.Path(__file__).parent)
    datasets_cfg = cfg["datasets"]

    parser = argparse.ArgumentParser(description=_DESC)
    parser.add_argument("-d", "--datasets", nargs="*",
                        help="The datasets to load/download and train models on. If empty, all datasets are used.")
    parser.add_argument("-m", "--models", nargs="*",
                        help="The models to train on each dataset. If empty, all models for that dataset's "
                             "problem type are used.")
    args = parser.parse_args()

    OUT_DIR = cfg["data_dir"]

    datasets = args.datasets if args.datasets is not None else list(datasets_cfg)
    prog = tqdm(datasets)

    for ds in prog:
        prog.set_postfix({"dataset": ds})
        prog.set_description("Loading data...")
        split = load(
            ds, OUT_DIR,
            pred_type=datasets_cfg[ds]["pred_type"],
            data_id=datasets_cfg[ds]["data_id"],
            download=True,
        )
        models = args.models if args.models is not None else model_names(split.pred_type)

        ds_dir = os.path.join(OUT_DIR, ds, "models")
        os.makedirs(ds_dir, exist_ok=True)
        with open(os.path.join(ds_dir, "scores.csv"), "w") as fp:
            writer = csv.DictWriter(fp, ["model", "score"])
            writer.writeheader()
            for model_name in models:
                prog.set_description(f"Training {model_name}...")
                model = make_model(split.pred_type, model_name)
                model.fit(split.X_train.to_numpy(), split.y_train)

                prog.set_description(f"Testing {model_name}...")
                y_pred = model.predict(split.X_test.to_numpy())
                score = (
                    balanced_accuracy_score(split.y_test, y_pred)
                    if split.pred_type == "classification"
                    else r2_score(split.y_test, y_pred)
                )
                writer.writerow({"model": model_name, "score": score})

                joblib.dump(model, os.path.join(ds_dir, f"{model_name}.pkl"))
