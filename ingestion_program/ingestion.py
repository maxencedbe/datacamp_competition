import argparse, os, time, traceback
import pandas as pd
import numpy as np
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir",       default="/app/input_data")
parser.add_argument("--output-dir",     default="/app/output")
parser.add_argument("--submission-dir", default="//app/ingested_program")
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

    # === 3. Import submission ===
    submission_dir = args.submission_dir
    print("[DEBUG] submission_dir: {}".format(submission_dir))
    print("[DEBUG] submission_dir exists: {}".format(os.path.exists(submission_dir)))
    if os.path.exists(submission_dir):
        print("[DEBUG] submission_dir contents: {}".format(os.listdir(submission_dir)))

    sys.path.insert(0, submission_dir)
    print("[DEBUG] Importing submission...")
    from submission import get_model
    print("[DEBUG] Import OK")

    # === 4. Train and predict ===
    start = time.time()
    model = get_model()
    print("[DEBUG] Model created, starting fit...")
    model.fit(X_train, y_train[label_cols], train_img_dir)
    print("[DEBUG] Fit done, predicting...")
    predictions = model.predict(X_test, test_img_dir)
    elapsed = time.time() - start
    print("[DEBUG] Predictions shape: {}".format(predictions.shape))

    # === 5. Save results ===
    pred_df = pd.DataFrame(predictions, columns=label_cols)
    pred_df.insert(0, "filename", X_test["filename"].values)
    pred_df.to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)

    pd.Series([elapsed], name="runtime").to_csv(
        os.path.join(args.output_dir, "runtime.csv"), index=False
    )
    print("Ingestion OK in {:.2f}s".format(elapsed))

except Exception as e:
    # Print the full error so it shows in CodaBench logs
    print("=" * 60)
    print("INGESTION ERROR")
    print("=" * 60)
    traceback.print_exc()
    print("=" * 60)

    # Write dummy predictions so scoring runs and shows the error
    try:
        X_test = pd.read_csv(os.path.join(data_dir, "test.csv"))
        n = len(X_test)
    except Exception:
        n = 1
    dummy = pd.DataFrame(
        np.zeros((n, 5)),
        columns=label_cols
    )
    dummy.insert(0, "filename", ["dummy"] * n)
    dummy.to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)
    pd.Series([-1], name="runtime").to_csv(
        os.path.join(args.output_dir, "runtime.csv"), index=False
    )
    print("Dummy predictions written so scoring can show this error.")