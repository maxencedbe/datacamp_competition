import argparse
import os
import sys
import time
import traceback
import importlib.util

import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir",       default="/app/input_data")
parser.add_argument("--output-dir",     default="/app/output")
parser.add_argument("--submission-dir", default="/app/ingested_program")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

label_cols = ["0", "1", "2", "3", "4"]

try:
    # === 1. Find the actual data directory ===
    data_dir = args.data_dir
    print("[DEBUG] Initial data_dir: {}".format(data_dir))
    print("[DEBUG] data_dir exists: {}".format(os.path.exists(data_dir)))
    if os.path.exists(data_dir):
        print("[DEBUG] Contents: {}".format(os.listdir(data_dir)))

    if not os.path.isfile(os.path.join(data_dir, "train.csv")):
        print("[DEBUG] train.csv not found, searching...")
        parent = os.path.dirname(data_dir)
        if os.path.exists(parent):
            print("[DEBUG] Parent ({}): {}".format(parent, os.listdir(parent)))
        for root, dirs, files in os.walk(parent):
            if "train.csv" in files:
                data_dir = root
                print("[DEBUG] Found train.csv in: {}".format(data_dir))
                break

    print("[DEBUG] Final data_dir: {}".format(data_dir))

    # === 2. Load data ===
    X_train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "train_labels.csv"))
    X_test  = pd.read_csv(os.path.join(data_dir, "test.csv"))
    print("[DEBUG] Data loaded: {} train, {} test".format(len(X_train), len(X_test)))

    train_img_dir = os.path.join(data_dir, "train_images")
    test_img_dir  = os.path.join(data_dir, "test_images")

    # === 3. Find and import submission.py ===
    submission_dir = args.submission_dir
    print("[DEBUG] submission_dir: {}".format(submission_dir))
    print("[DEBUG] submission_dir exists: {}".format(os.path.exists(submission_dir)))
    if os.path.exists(submission_dir):
        print("[DEBUG] submission_dir contents: {}".format(os.listdir(submission_dir)))

    # Search for submission.py in submission_dir and subdirectories
    submission_path = None
    if os.path.exists(submission_dir):
        for root, dirs, files in os.walk(submission_dir):
            if "submission.py" in files:
                submission_path = os.path.join(root, "submission.py")
                print("[DEBUG] Found submission.py at: {}".format(submission_path))
                break

    # Also try common CodaBench mount points
    if submission_path is None:
        for extra in ["/app/program", "/app/ingested_program", "/app/submission"]:
            if os.path.exists(extra):
                for root, dirs, files in os.walk(extra):
                    if "submission.py" in files:
                        submission_path = os.path.join(root, "submission.py")
                        print("[DEBUG] Found submission.py at: {}".format(submission_path))
                        break
            if submission_path is not None:
                break

    if submission_path is None:
        raise FileNotFoundError(
            "submission.py not found anywhere. submission_dir={}".format(args.submission_dir)
        )

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("submission", submission_path)
    submission_mod = importlib.util.module_from_spec(spec)
    sys.modules["submission"] = submission_mod
    spec.loader.exec_module(submission_mod)
    get_model = submission_mod.get_model
    print("[DEBUG] submission.py imported successfully")

    # === 4. Train and predict ===
    start = time.time()
    model = get_model()
    print("[DEBUG] Model created, starting fit...")
    model.fit(X_train, y_train[label_cols], train_img_dir)
    print("[DEBUG] Fit done, starting predict...")
    predictions = model.predict(X_test, test_img_dir)
    elapsed = time.time() - start
    print("[DEBUG] Predict done in {:.2f}s".format(elapsed))

    # === 5. Save results ===
    pred_df = pd.DataFrame(predictions, columns=label_cols)
    pred_df.insert(0, "filename", X_test["filename"].values)
    pred_df.to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)

    pd.Series([elapsed], name="runtime").to_csv(
        os.path.join(args.output_dir, "runtime.csv"), index=False
    )
    print("Ingestion OK in {:.2f}s".format(elapsed))

except Exception as e:
    print("=" * 60)
    print("INGESTION ERROR")
    print("=" * 60)
    traceback.print_exc()
    print("=" * 60)

    # Write dummy predictions so scoring still runs
    try:
        X_test = pd.read_csv(os.path.join(data_dir, "test.csv"))
        n = len(X_test)
        fnames = X_test["filename"].values.tolist()
    except Exception:
        n = 1
        fnames = ["dummy"]
    dummy = pd.DataFrame(np.zeros((n, 5)), columns=label_cols)
    dummy.insert(0, "filename", fnames)
    dummy.to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)
    pd.Series([-1], name="runtime").to_csv(
        os.path.join(args.output_dir, "runtime.csv"), index=False
    )
    print("Dummy predictions written.")