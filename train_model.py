import os
import numpy as np
import pandas as pd
from tensorflow.keras import layers
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from dataset_utils import df_to_dataset

# ✅ PATHS
DATA_DIR = "./sf/EuroSAT/"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
VAL_CSV = os.path.join(DATA_DIR, "validation.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

# ✅ SETTINGS
IMG_SIZE = (64, 64)
BATCH = 32
EPOCHS = 10

# ✅ SAVE DIRECTORY
SAVEDIR = "./saved_models"
os.makedirs(SAVEDIR, exist_ok=True)

# ✅ LOAD ORIGINAL CSV
train_df = pd.read_csv(TRAIN_CSV)

# ✅ Extract class names from CSV
class_names = sorted(train_df["ClassName"].unique())

# ✅ Prepare Label Encoder
le = LabelEncoder()
le.fit(class_names)

# ✅ Convert original CSV → new CSV format (image_path, label)
def convert_csv_format(old_csv_path, new_csv_path):
    df = pd.read_csv(old_csv_path)

    df_new = pd.DataFrame()
    df_new["image_path"] = df["Filename"]
    df_new["label"] = df["ClassName"]

    df_new.to_csv(new_csv_path, index=False)

# ✅ Convert all 3 CSV files
convert_csv_format(TRAIN_CSV, "train_fixed.csv")
convert_csv_format(VAL_CSV, "validation_fixed.csv")
convert_csv_format(TEST_CSV, "test_fixed.csv")

# ✅ Load datasets using fixed CSVs
train_ds = df_to_dataset(DATA_DIR, "train_fixed.csv", le, IMG_SIZE, BATCH, shuffle=True)
val_ds = df_to_dataset(DATA_DIR, "validation_fixed.csv", le, IMG_SIZE, BATCH, shuffle=False)
test_ds = df_to_dataset(DATA_DIR, "test_fixed.csv", le, IMG_SIZE, BATCH, shuffle=False)

# ✅ Build Transfer Learning Model (MobileNetV2)
base = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)
base.trainable = False  # freeze base layers

model = tf.keras.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ✅ Model Summary
model.summary()

# ✅ Train model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ✅ Save Model + Class Names
model.save(os.path.join(SAVEDIR, "eurosat_model.keras"))
np.save(os.path.join(SAVEDIR, "class_names.npy"), np.array(class_names))

print("\n✅ Training Completed Successfully!")
print("✅ Model saved to: saved_models/eurosat_model.keras")
print("✅ Class names saved to: saved_models/class_names.npy")
