import argparse, os, time
import pandas as pd
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir",       default="/app/input_data")
parser.add_argument("--output-dir",     default="/app/output")
parser.add_argument("--submission-dir", default="/app/ingested_program")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# Charger les données
X_train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
y_train = pd.read_csv(os.path.join(args.data_dir, "train_labels.csv"))
X_test  = pd.read_csv(os.path.join(args.data_dir, "test.csv"))

train_img_dir = os.path.join(args.data_dir, "train_images")
test_img_dir  = os.path.join(args.data_dir, "test_images")

sys.path.insert(0, args.submission_dir)
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