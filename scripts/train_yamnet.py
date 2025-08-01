import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration 
DATA_DIR = Path("features")
MODEL_OUT_PATH = Path("regressor_model.keras")
PLOT_OUT_PATH = Path("training_loss.png")

# Load Data 
X = np.load(DATA_DIR / "X_embeddings.npy")
y = np.load(DATA_DIR / "y_val_ar.npy")

# Train/Val Split 
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Build Model 
model = keras.Sequential([
    layers.Input(shape=(1024,)),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(2, activation='linear')  # valence, arousal
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# ---- Train ----
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=128,
    callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
)

# ---- Save Model ----
model.save(MODEL_OUT_PATH)

# ---- Plot Training Loss ----
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Training Loss')
plt.savefig(PLOT_OUT_PATH)
plt.close()

print(f"✅ Model saved to {MODEL_OUT_PATH}")