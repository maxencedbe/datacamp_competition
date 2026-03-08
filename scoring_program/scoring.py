import argparse, os, json
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, hamming_loss

parser = argparse.ArgumentParser()
parser.add_argument("--reference-dir",  default="/app/input/ref")
parser.add_argument("--prediction-dir", default="/app/input/res")
parser.add_argument("--output-dir",     default="/app/output")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

label_cols = ["0", "1", "2", "3", "4"]

y_true = pd.read_csv(os.path.join(args.reference_dir, "test_labels.csv"))[label_cols]
y_pred = pd.read_csv(os.path.join(args.prediction_dir, "predictions.csv"))[label_cols]

y_pred_bin = (y_pred >= 0.5).astype(int)

import warnings
warnings.filterwarnings("ignore")

try:
    scores = {
        "f1_macro":    round(f1_score(y_true, y_pred_bin, average="macro"), 4),
        "f1_micro":    round(f1_score(y_true, y_pred_bin, average="micro"), 4),
        "hamming_loss": round(hamming_loss(y_true, y_pred_bin), 4),
    }
except Exception as e:
    print("Scoring error: {}".format(e))
    scores = {"f1_macro": 0, "f1_micro": 0, "hamming_loss": 1}


try:
    runtime = pd.read_csv(
        os.path.join(args.prediction_dir, "runtime.csv")
    ).squeeze().item()
    scores["runtime"] = round(runtime, 3)
except Exception:
    scores["runtime"] = -1

print("Scores :", scores)
with open(os.path.join(args.output_dir, "scores.json"), "w") as f:
    json.dump(scores, f)