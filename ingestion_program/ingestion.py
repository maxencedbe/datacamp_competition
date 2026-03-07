import argparse, os, time, glob
import pandas as pd
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir",       default="/app/input_data")
parser.add_argument("--output-dir",     default="/app/output")
parser.add_argument("--submission-dir", default="/app/ingested_program")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# === Debug: find the actual data directory ===
data_dir = args.data_dir
print(f"[DEBUG] Initial data_dir: {data_dir}")
print(f"[DEBUG] data_dir exists: {os.path.exists(data_dir)}")
if os.path.exists(data_dir):
    print(f"[DEBUG] Contents of data_dir: {os.listdir(data_dir)}")

# If train.csv is not directly in data_dir, search subdirectories
if not os.path.isfile(os.path.join(data_dir, "train.csv")):
    print(f"[DEBUG] train.csv not found in {data_dir}, searching subdirs...")
    # Try parent directory
    parent = os.path.dirname(data_dir)
    if os.path.exists(parent):
        print(f"[DEBUG] Parent dir ({parent}): {os.listdir(parent)}")
    # Search for train.csv recursively
    for root, dirs, files in os.walk(os.path.dirname(data_dir)):
        print(f"[DEBUG] Scanning {root}: dirs={dirs}, files={files[:5]}")
        if "train.csv" in files:
            data_dir = root
            print(f"[DEBUG] Found train.csv in: {data_dir}")
            break
    else:
        # Also try common CodaBench mount patterns
        for candidate in [
            data_dir,
            os.path.join(data_dir, "data"),
            os.path.join(data_dir, "input_data"),
            data_dir + "_data",
            data_dir.replace("input_data", "input_data_data"),
        ]:
            if os.path.isfile(os.path.join(candidate, "train.csv")):
                data_dir = candidate
                print(f"[DEBUG] Found train.csv in candidate: {data_dir}")
                break

print(f"[DEBUG] Final data_dir: {data_dir}")
print(f"[DEBUG] Contents: {os.listdir(data_dir) if os.path.exists(data_dir) else 'DIR NOT FOUND'}")

# Charger les données
X_train = pd.read_csv(os.path.join(data_dir, "train.csv"))
y_train = pd.read_csv(os.path.join(data_dir, "train_labels.csv"))
X_test  = pd.read_csv(os.path.join(data_dir, "test.csv"))

train_img_dir = os.path.join(data_dir, "train_images")
test_img_dir  = os.path.join(data_dir, "test_images")

# Locate submission.py in the submission directory (may be in a subdirectory)
submission_dir = args.submission_dir
print(f"[DEBUG] submission_dir: {submission_dir}")
print(f"[DEBUG] submission_dir exists: {os.path.exists(submission_dir)}")
if os.path.exists(submission_dir):
    print(f"[DEBUG] Contents of submission_dir: {os.listdir(submission_dir)}")

if not os.path.isfile(os.path.join(submission_dir, "submission.py")):
    # Search subdirectories for submission.py
    for root, dirs, files in os.walk(submission_dir):
        print(f"[DEBUG] Scanning {root}: files={files}")
        if "submission.py" in files:
            submission_dir = root
            print(f"[DEBUG] Found submission.py in: {submission_dir}")
            break

sys.path.insert(0, submission_dir)
from submission import get_model

label_cols = ["0", "1", "2", "3", "4"]

start = time.time()
model = get_model()
model.fit(X_train, y_train[label_cols], train_img_dir)
predictions = model.predict(X_test, test_img_dir)
elapsed = time.time() - start

# Sauvegarder
pred_df = pd.DataFrame(predictions, columns=label_cols)
pred_df.insert(0, "filename", X_test["filename"].values)
pred_df.to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)

pd.Series([elapsed], name="runtime").to_csv(
    os.path.join(args.output_dir, "runtime.csv"), index=False
)
print(f"Ingestion terminée en {elapsed:.2f}s")