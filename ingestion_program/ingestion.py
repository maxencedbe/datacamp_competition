import subprocess, sys
import argparse, os, time, glob
import importlib.util

# Install dependencies compatible with Python 3.7
try:
    import torch
    print(f"[DEBUG] torch already installed: {torch.__version__}")
except ImportError:
    print("[DEBUG] Installing PyTorch for Python 3.7...")
    # Pin typing-extensions to last Python 3.7 compatible version
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "typing-extensions==4.5.0",
    ])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "torch==1.13.1+cu117", "torchvision==0.14.1+cu117",
        "--extra-index-url", "https://download.pytorch.org/whl/cu117",
    ])

for pkg in ["pandas", "scikit-learn", "Pillow"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--quiet", pkg,
        ])

import pandas as pd

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

# Locate and load submission.py
submission_dir = args.submission_dir
print(f"[DEBUG] submission_dir: {submission_dir}")
print(f"[DEBUG] submission_dir exists: {os.path.exists(submission_dir)}")

# List all candidate paths for submission.py
candidates = [submission_dir]
if os.path.exists(submission_dir):
    print(f"[DEBUG] Contents of submission_dir: {os.listdir(submission_dir)}")
    for root, dirs, files in os.walk(submission_dir):
        if root != submission_dir:
            candidates.append(root)

# Also try common CodaBench mount points
for extra in ["/app/program", "/app/ingested_program", "/app/submission"]:
    if extra not in candidates and os.path.exists(extra):
        candidates.append(extra)
        for root, dirs, files in os.walk(extra):
            if root != extra:
                candidates.append(root)

# Find submission.py
submission_path = None
for d in candidates:
    p = os.path.join(d, "submission.py")
    if os.path.isfile(p):
        submission_path = p
        print(f"[DEBUG] Found submission.py at: {p}")
        break

if submission_path is None:
    # Last resort: recursive search from /app
    print("[DEBUG] Searching /app recursively for submission.py...")
    for root, dirs, files in os.walk("/app"):
        print(f"[DEBUG]   {root}: {files[:10]}")
        if "submission.py" in files:
            submission_path = os.path.join(root, "submission.py")
            print(f"[DEBUG] Found submission.py at: {submission_path}")
            break

if submission_path is None:
    raise FileNotFoundError(
        f"submission.py not found. submission_dir={args.submission_dir}"
    )

# Load the module dynamically using importlib
spec = importlib.util.spec_from_file_location("submission", submission_path)
submission = importlib.util.module_from_spec(spec)
sys.modules["submission"] = submission
spec.loader.exec_module(submission)
get_model = submission.get_model

label_cols = ["0", "1", "2", "3", "4"]

print(f"[DEBUG] output_dir: {args.output_dir}")
print(f"[DEBUG] output_dir exists: {os.path.exists(args.output_dir)}")

try:
    start = time.time()
    model = get_model()
    print("[DEBUG] Model created, starting fit...")
    model.fit(X_train, y_train[label_cols], train_img_dir)
    print("[DEBUG] Fit done, starting predict...")
    predictions = model.predict(X_test, test_img_dir)
    elapsed = time.time() - start
    print(f"[DEBUG] Predict done in {elapsed:.2f}s")
except Exception as e:
    print(f"[ERROR] Model fit/predict failed: {e}")
    import traceback
    traceback.print_exc()
    # Write dummy predictions so scoring can at least run
    elapsed = 0
    predictions = pd.DataFrame(
        0.0, index=range(len(X_test)), columns=label_cols
    ).values

# Sauvegarder
pred_df = pd.DataFrame(predictions, columns=label_cols)
pred_df.insert(0, "filename", X_test["filename"].values)
out_path = os.path.join(args.output_dir, "predictions.csv")
pred_df.to_csv(out_path, index=False)
print(f"[DEBUG] predictions.csv written to: {out_path}")
print(f"[DEBUG] File exists: {os.path.isfile(out_path)}")
print(f"[DEBUG] output_dir contents: {os.listdir(args.output_dir)}")

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