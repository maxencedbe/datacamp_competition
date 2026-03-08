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
print(f"Ingestion terminée en {elapsed:.2f}s")