import argparse
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from experiments.real.datasets import get_valid_datasets, get_dataset, get_pred_type
from tqdm import tqdm
import csv
import os
import joblib

_DESC = """
This script downloads datasets, splits them into train and test set, 
and trains and saves one or multiple models on them.

Each dataset is saved to ./data/[DATASET_NAME]. This folder contains the full dataset
(data.csv, labels.csv), as well as a pre-made train test split (X_train.csv, y_train.csv, X_test.csv, y_test.csv).

The models are saved to ./data/[DATASET_NAME]/models. Each model is pickled to [MODEL_NAME].pkl.
The models directory also contains a scores.csv file, which contains the balanced accuracy (in case of classification)
or R^2 (in case of regression) for each model.
"""

_MODEL_DICT = {
    "classification": {
        "knn": KNeighborsClassifier, "gradientboosting": GradientBoostingClassifier
    },
    "regression": {
        "knn": KNeighborsRegressor, "gradientboosting": GradientBoostingRegressor
    }
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=_DESC)
    parser.add_argument("-d", "--datasets", nargs="*",
                        help="The datasets to load/download and train models on. If empty, all datasets are used.")
    parser.add_argument("-m", "--models", nargs="*",
                        help="The models to train on each dataset. If empty, all models are used.")
    args = parser.parse_args()

    OUT_DIR = "./data"

    datasets = args.datasets if args.datasets is not None else get_valid_datasets()
    models = args.models if args.models is not None else ["knn", "gradientboosting"]
    prog = tqdm(datasets)

    for ds in prog:
        prog.set_postfix({"dataset": ds})
        prog.set_description("Loading data...")
        X_train, X_test, y_train, y_test = get_dataset(ds, OUT_DIR, download=True)
        pred_type = get_pred_type(ds)
        if len(y_train.shape) == 2 and y_test.shape[1] == 1:
            y_train = y_train.ravel()
            y_test = y_test.ravel()

        ds_dir = os.path.join(OUT_DIR, ds, "models")
        os.makedirs(ds_dir, exist_ok=True)
        with open(os.path.join(ds_dir, "scores.csv"), "w") as fp:
            writer = csv.DictWriter(fp, ["model", "score"])
            writer.writeheader()
            for model_name in models:
                prog.set_description(f"Training {model_name}...")
                model = _MODEL_DICT[pred_type][model_name]()
                model.fit(X_train.to_numpy(), y_train)

                prog.set_description(f"Testing {model_name}...")
                y_pred = model.predict(X_test.to_numpy())
                score = balanced_accuracy_score(y_test, y_pred) if pred_type == "classification" else r2_score(y_test, y_pred)
                writer.writerow({"model": model_name, "score": score})

                joblib.dump(model, os.path.join(ds_dir, f"{model_name}.pkl"))
