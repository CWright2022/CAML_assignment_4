#!/usr/bin/env python3

import argparse
import json
import time
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder  

# --- Load main.py as a module ------------------------------------------------
MAIN_PY = Path("main.py")  
spec = importlib.util.spec_from_file_location("main", str(MAIN_PY))
main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main)

def prepare_stage1_data(root, split):
    """
    Runs the same data-loading + Stage 0 pipeline as main.main, but returns the
    Stage 1 matrices so we can reuse them for tuning.

    Parameters
    ----------
    root  : str
        Path to dataset root (e.g., './iot_data').
    split : float
        Train/test split ratio (same semantics as main.main: fraction of samples
        used for training).

    Returns
    -------
    X_tr_full : np.ndarray
        Stage 1 training features (flow features + NB outputs).
    X_ts_full : np.ndarray
        Stage 1 testing features (flow features + NB outputs).
    Y_tr      : np.ndarray
        Encoded training labels.
    Y_ts      : np.ndarray
        Encoded testing labels.
    """
    print("Loading dataset (from main.load_data) ...")
    X, X_p, X_d, X_c, Y = main.load_data(root)

    # Encode labels (same as in main.py)
    print("Encoding labels ...")
    le = LabelEncoder()
    Y = le.fit_transform(Y)

    # Shuffle using same seed as main.py for reproducibility
    SEED = getattr(main, "SEED", 0)
    print(f"Shuffling dataset using seed {SEED} ...")
    Y = np.asarray(Y)
    s = np.arange(Y.shape[0])
    np.random.seed(SEED)
    np.random.shuffle(s)
    X, X_p, X_d, X_c, Y = X[s], X_p[s], X_d[s], X_c[s], Y[s]

    # Split train/test the same way as main.main
    print(f"Splitting dataset using train:test ratio of {int(split*10)}:{int((1-split)*10)} ...")
    cut = int(len(Y) * split)
    X_tr, Xp_tr, Xd_tr, Xc_tr, Y_tr = X[cut:], X_p[cut:], X_d[cut:], X_c[cut:], Y[cut:]
    X_ts, Xp_ts, Xd_ts, Xc_ts, Y_ts = X[:cut], X_p[:cut], X_d[:cut], X_c[:cut], Y[:cut]

    # Stage 0: Naive Bayes on ports/domains/ciphers (exactly like main.do_stage_0)
    print("Running Stage 0 (Naive Bayes on multinomial features) ...")
    p_tr, p_ts, d_tr, d_ts, c_tr, c_ts = main.do_stage_0(
        Xp_tr, Xp_ts, Xd_tr, Xd_ts, Xc_tr, Xc_ts, Y_tr, Y_ts
    )

    # Build Stage 1 feature matrices: flow features + NB outputs
    print("Building Stage 1 feature matrices ...")
    X_tr_full = np.hstack((X_tr, p_tr, d_tr, c_tr))
    X_ts_full = np.hstack((X_ts, p_ts, d_ts, c_ts))

    print("Stage 1 data shapes:")
    print("  X_tr_full:", X_tr_full.shape)
    print("  X_ts_full:", X_ts_full.shape)

    return X_tr_full, X_ts_full, Y_tr, Y_ts


def run_with_params(X_tr, X_ts, Y_tr, Y_ts, params):
    """
    Train and evaluate a RandomForest from main.py with the given params.

    Parameters
    ----------
    X_tr, X_ts : np.ndarray
        Stage 1 training/testing features.
    Y_tr, Y_ts : np.ndarray
        Training/testing labels (encoded as ints).
    params : dict
        Parameter dictionary with keys:
          - "maxDepth"
          - "minNode"
          - "numTrees"
          - "dataFraction"
          - "featureSubcount"  (string: "sqrt", "half", or None)

    Returns
    -------
    metrics : dict
        Contains accuracy, train_time_s, predict_time_s.
    """
    metrics = {}

    # Resolve feature_subcount from the string option
    n_features = X_tr.shape[1]
    if params["featureSubcount"] == "sqrt":
        feature_subcount = max(1, int(np.sqrt(n_features)))
    elif params["featureSubcount"] == "half":
        feature_subcount = max(1, n_features // 2)
    else:
        feature_subcount = None

    # Instantiate custom RandomForest 
    rf = main.RandomForest(
        num_trees = params["numTrees"],
        max_depth = params["maxDepth"],
        min_node = params["minNode"],
        data_fraction = params["dataFraction"],
        feature_subcount = feature_subcount,
    )

    # Fit and time training
    t0 = time.time()
    rf.fit(X_tr, Y_tr)
    train_time = time.time() - t0

    # Predict and time inference
    t1 = time.time()
    pred = rf.predict(X_ts)
    pred_time = time.time() - t1

    accuracy = float((pred == Y_ts).mean())

    metrics.update({
        "accuracy": accuracy,
        "train_time_s": train_time,
        "predict_time_s": pred_time,
    })
    return metrics

def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--root",
        required=True,
        help="Path to data folder (same as main.py expects, e.g. ./iot_data)."
    )
    parser.add_argument(
        "-s", "--split",
        type=float,
        default=0.7,
        help="Train/test split ratio (fraction used for training)."
    )
    parser.add_argument(
        "--outdir",
        default="tuning_output",
        help="Directory in which to store tuning results."
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Prepare Stage 1 dataset once and reuse for all hyperparameter combinations
    X_tr_full, X_ts_full, Y_tr, Y_ts = prepare_stage1_data(args.root, args.split)

    # Fixed tuning search grid 
    DEPTHS = [10, 14, 18]
    MIN_NODES = [4, 8]
    N_TREES = [50, 100]
    DATA_FRACS = [0.6, 0.7]
    FEATURE_SUBCOUNTS = ["sqrt", "half"]

    results = []
    best = None
    best_acc = -1.0

    for md in DEPTHS:
        for mn in MIN_NODES:
            for nt in N_TREES:
                for df in DATA_FRACS:
                    for fs in FEATURE_SUBCOUNTS:
                        params = {
                            "maxDepth": md,
                            "minNode": mn,
                            "numTrees": nt,
                            "dataFraction": df,
                            "featureSubcount": fs,
                        }

                        print(f"\n Testing Parameters: {params}")
                        metrics = run_with_params(X_tr_full, X_ts_full, Y_tr, Y_ts, params)
                        row = {**params, **metrics}
                        results.append(row)

                        if metrics["accuracy"] > best_acc:
                            best_acc = metrics["accuracy"]
                            best = row

    # Save full results and best parameters
    df = pd.DataFrame(results)
    df.to_csv(outdir / "tuning_results.csv", index=False)
    with open(outdir / "best_params.json", "w") as fp:
        json.dump(best, fp, indent=2)

    print("\nTUNING COMPLETE")
    print("Best parameters:")
    print(json.dumps(best, indent = 2))
    print(f"\nResults saved to: {outdir}/")


if __name__ == "__main__":
    main_cli()
