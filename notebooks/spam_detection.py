"""
SMS Spam Detection — Complete Training Pipeline
================================================
Run from the project root:
    python -X utf8 notebooks/spam_detection.py

Dataset : SMS Spam Collection (UCI)
Model   : Embedding -> LSTM -> Dropout -> Sigmoid
TF ver. : 2.x  (tested on 2.21)
"""

# ── 0. Fix Windows UTF-8 console ──────────────────────────────────────────────
import sys, io
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout, Input
)
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

print("=" * 60)
print("  SMS Spam Detection  -  Deep Learning Project")
print("=" * 60)
print(f"TensorFlow : {tf.__version__}")
print(f"NumPy      : {np.__version__}")
print()

# ── 1. Paths ──────────────────────────────────────────────────────────────────
ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(ROOT, "data", "SMSSpamCollection")
MODEL_DIR = os.path.join(ROOT, "models")
IMG_DIR   = os.path.join(ROOT, "notebooks", "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMG_DIR,   exist_ok=True)

# ── 2. Load dataset ───────────────────────────────────────────────────────────
print("[1/7] Loading dataset...")

if not os.path.exists(DATA_PATH):
    print()
    print("  Dataset not found at:", DATA_PATH)
    print("  Download from: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection")
    print("  Place 'SMSSpamCollection' in the data/ folder, then re-run.")
    sys.exit(1)

df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label", "message"])
print(f"  Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")
print(df.head().to_string())
print()

# ── 3. Explore ────────────────────────────────────────────────────────────────
print("[2/7] Exploring data...")
print(df["label"].value_counts().to_string())
print()

# Bar chart
fig, ax = plt.subplots(figsize=(5, 4))
counts  = df["label"].value_counts()
labels  = counts.index.tolist()
values  = counts.values.tolist()
colors  = ["#10b981", "#ef4444"]
bars    = ax.bar(labels, values, color=colors, edgecolor="none", width=0.5)
ax.set_title("Spam vs Ham Distribution", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Message Type")
ax.set_ylabel("Count")
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 30,
            str(val), ha="center", va="bottom", fontweight="bold", color="white")
ax.set_facecolor("#1e1b4b")
fig.patch.set_facecolor("#1e1b4b")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.tick_params(colors="white")
ax.yaxis.label.set_color("white")
ax.xaxis.label.set_color("white")
ax.title.set_color("white")
plt.tight_layout()
dist_path = os.path.join(IMG_DIR, "class_distribution.png")
plt.savefig(dist_path, dpi=120, facecolor=fig.get_facecolor())
plt.close()
print(f"  Class distribution chart -> {dist_path}")

# Message length stats
df["msg_len"] = df["message"].apply(len)
print(df.groupby("label")["msg_len"].describe().to_string())
print()

# ── 4. Encode labels ──────────────────────────────────────────────────────────
df["label"] = df["label"].map({"ham": 0, "spam": 1})
print("[3/7] Labels encoded  ham=0, spam=1")
print(df["label"].value_counts().to_string())
print()

# ── 5. Train / test split ─────────────────────────────────────────────────────
# Convert to plain Python lists to avoid pyarrow/numpy indexing conflicts
X = df["message"].tolist()
y = df["label"].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train = np.array(X_train)
X_test  = np.array(X_test)
y_train = np.array(y_train)
y_test  = np.array(y_test)
print(f"[4/7] Split  train={len(X_train)}  test={len(X_test)}")
print()

# ── 6. Text vectorization (applied outside the model) ─────────────────────────
MAX_TOKENS  = 10_000
MAX_LEN     = 100

print("[5/7] Vectorizing text...")
vectorizer = TextVectorization(
    max_tokens=MAX_TOKENS,
    output_mode="int",
    output_sequence_length=MAX_LEN,
    standardize="lower_and_strip_punctuation",
)
vectorizer.adapt(X_train)

print(f"  Vocabulary size: {len(vectorizer.get_vocabulary())}")

X_train_vec = vectorizer(X_train).numpy()
X_test_vec  = vectorizer(X_test).numpy()
print(f"  X_train shape : {X_train_vec.shape}")
print(f"  X_test  shape : {X_test_vec.shape}")
print()

# Save vocabulary to a plain text file (most reliable cross-session approach)
vocab     = vectorizer.get_vocabulary()
vec_path  = os.path.join(MODEL_DIR, "vocab.txt")
with open(vec_path, "w", encoding="utf-8") as f:
    for word in vocab:
        f.write(word + "\n")

# Also save hyperparams so the app can rebuild the vectorizer
vec_meta  = {"max_tokens": MAX_TOKENS, "max_len": MAX_LEN, "vocab_size": len(vocab)}
meta_path = os.path.join(MODEL_DIR, "vec_meta.pkl")
with open(meta_path, "wb") as f:
    pickle.dump(vec_meta, f)

print(f"  Vocabulary  ({len(vocab)} tokens) -> {vec_path}")
print(f"  Vec meta    -> {meta_path}")
print()

# ── 7. Build LSTM model ───────────────────────────────────────────────────────
print("[6/7] Building LSTM model...")

model = Sequential([
    Input(shape=(MAX_LEN,), dtype="int32"),
    Embedding(input_dim=MAX_TOKENS + 1, output_dim=64, mask_zero=True),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid"),
], name="SMS_Spam_LSTM")

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.summary()
print()

# ── 8. Train ──────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(MODEL_DIR, "spam_model.keras")

callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=3,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint(filepath=MODEL_PATH, monitor="val_accuracy",
                    save_best_only=True, verbose=1),
]

print("Training... (up to 15 epochs with early stopping)")
history = model.fit(
    X_train_vec, y_train,
    epochs=15,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1,
)
print()
print(f"  Best model saved -> {MODEL_PATH}")
print()

# ── 9. Evaluate ───────────────────────────────────────────────────────────────
print("[7/7] Evaluating on test set...")
loss, accuracy = model.evaluate(X_test_vec, y_test, verbose=0)
print(f"  Test Loss     : {loss:.4f}")
print(f"  Test Accuracy : {accuracy * 100:.2f}%")
print()

preds       = model.predict(X_test_vec, verbose=0)
pred_labels = (preds > 0.5).astype(int).flatten()

print("Classification Report:")
print(classification_report(y_test, pred_labels, target_names=["Ham", "Spam"]))

# ── 10. Confusion matrix ──────────────────────────────────────────────────────
cm = confusion_matrix(y_test, pred_labels)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Ham", "Spam"],
            yticklabels=["Ham", "Spam"],
            ax=ax, linewidths=0.5)
ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold", pad=10)
ax.set_ylabel("True Label")
ax.set_xlabel("Predicted Label")
plt.tight_layout()
cm_path = os.path.join(IMG_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=120)
plt.close()
print(f"  Confusion matrix  -> {cm_path}")

# ── 11. Training curves ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(history.history["accuracy"],     label="Train",
             color="#a78bfa", linewidth=2)
axes[0].plot(history.history["val_accuracy"], label="Validation",
             color="#60a5fa", linewidth=2, linestyle="--")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history.history["loss"],     label="Train",
             color="#f472b6", linewidth=2)
axes[1].plot(history.history["val_loss"], label="Validation",
             color="#fb923c", linewidth=2, linestyle="--")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle("Training Performance", fontsize=14, fontweight="bold")
plt.tight_layout()
curves_path = os.path.join(IMG_DIR, "training_curves.png")
plt.savefig(curves_path, dpi=120)
plt.close()
print(f"  Training curves   -> {curves_path}")
print()

# ── 12. Custom message predictions ───────────────────────────────────────────
test_messages = [
    "Congratulations! You have won a FREE prize. Call now!",
    "Hey, are you coming to class tomorrow?",
    "URGENT! You have won $50,000. Click this link immediately!",
    "Can you send me the notes from today's lecture?",
    "FREE entry to a weekly competition. Text WIN to 80085!",
    "Don't forget mum's birthday is this weekend.",
    "Claim your FREE Nokia N70 mobile immediately. Call 09063000000",
    "Are you free to chat later tonight?",
]

print("=" * 60)
print("  Custom Message Predictions")
print("=" * 60)
vecs_custom   = vectorizer(np.array(test_messages)).numpy()
preds_custom  = model.predict(vecs_custom, verbose=0)

for msg, prob in zip(test_messages, preds_custom):
    tag = "[SPAM]    " if prob[0] > 0.5 else "[NOT SPAM]"
    print(f"\n  {tag} ({prob[0]*100:.1f}% spam probability)")
    print(f"  \"{msg[:80]}\"")

print()
print("=" * 60)
print("  Training complete!")
print(f"  Model      -> {MODEL_PATH}")
print(f"  Vectorizer -> {vec_path}")
print()
print("  To launch the web app, run:")
print("    streamlit run app/app.py")
print("=" * 60)
