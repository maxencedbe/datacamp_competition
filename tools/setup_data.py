import pandas as pd
import shutil
import os

# === PARAMÈTRES — adapte ces chemins ===
TRAIN_DIR = r"dataset\train"
TEST_DIR  = r"dataset\test"
CLASSES_FILE = "_classes.csv"

# === Chargement ===
train_df = pd.read_csv(os.path.join(TRAIN_DIR, CLASSES_FILE))
test_df  = pd.read_csv(os.path.join(TEST_DIR,  CLASSES_FILE))

# Nettoyer les noms de colonnes (espaces éventuels)
train_df.columns = train_df.columns.str.strip()
test_df.columns  = test_df.columns.str.strip()

label_cols = ["0", "1", "2", "3", "4"]

# === Séparer features (filename) et labels ===
X_train = train_df[["filename"]]
y_train = train_df[["filename"] + label_cols]

X_test  = test_df[["filename"]]
y_test  = test_df[["filename"] + label_cols]

# === Moitié public / moitié private ===
mid = len(X_test) // 2
X_test_public   = X_test.iloc[:mid]
y_test_public   = y_test.iloc[:mid]
X_test_private  = X_test.iloc[mid:]
y_test_private  = y_test.iloc[mid:]

# === Créer les dossiers ===
for path in [
    "dev_phase/input_data/train_images",
    "dev_phase/input_data/test_images",
    "dev_phase/reference_data",
    "test_phase/input_data/train_images",
    "test_phase/input_data/test_images",
    "test_phase/reference_data",
]:
    os.makedirs(path, exist_ok=True)

# === Sauvegarder les CSV ===
X_train.to_csv("dev_phase/input_data/train.csv", index=False)
y_train.to_csv("dev_phase/input_data/train_labels.csv", index=False)
X_test_public.to_csv("dev_phase/input_data/test.csv", index=False)
y_test_public.to_csv("dev_phase/reference_data/test_labels.csv", index=False)

X_train.to_csv("test_phase/input_data/train.csv", index=False)
y_train.to_csv("test_phase/input_data/train_labels.csv", index=False)
X_test_private.to_csv("test_phase/input_data/test.csv", index=False)
y_test_private.to_csv("test_phase/reference_data/test_labels.csv", index=False)

# === Copier les images ===
print("Copie des images train...")
for fname in train_df["filename"]:
    src = os.path.join(TRAIN_DIR, fname)
    for dest_dir in ["dev_phase/input_data/train_images", "test_phase/input_data/train_images"]:
        if os.path.exists(src):
            shutil.copy(src, os.path.join(dest_dir, fname))

print("Copie des images test publiques...")
for fname in X_test_public["filename"]:
    src = os.path.join(TEST_DIR, fname)
    if os.path.exists(src):
        shutil.copy(src, "dev_phase/input_data/test_images/")

print("Copie des images test privées...")
for fname in X_test_private["filename"]:
    src = os.path.join(TEST_DIR, fname)
    if os.path.exists(src):
        shutil.copy(src, "test_phase/input_data/test_images/")

print(f"Train : {len(train_df)} images")
print(f"Test public  : {len(X_test_public)} images")
print(f"Test private : {len(X_test_private)} images")
print("Données générées !")