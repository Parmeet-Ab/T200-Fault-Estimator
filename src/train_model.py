import numpy as np 
import tensorflow as tf
from tensorflow import keras


data = np.load("data/t200_dataset.npz")
X_normal = data["X_normal"]
X_fault = data["X_fault"]

mean = np.mean(X_normal, axis=0)
std = np.std(X_normal, axis=0)

X_normal = (X_normal - mean) / std
X_fault = (X_fault - mean) / std

model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(X_normal.shape[1],)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(16, activation="relu"),

    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(X_normal.shape[1])
])

model.compile(
    optimizer="adam",
    loss="mse"
)

model.fit(X_normal, X_normal, epochs=30, batch_size=64)

recon_normal = model.predict(X_normal)
recon_fault = model.predict(X_fault)

error_normal = np.mean((X_normal - recon_normal)**2, axis=1)
error_fault = np.mean((X_fault - recon_fault)**2, axis=1)

print("Normal error:", error_normal.mean())
print("Fault error:", error_fault.mean())

#save the file
import os
os.makedirs("models", exist_ok=True)
model.save("models/t200_autoencoder.keras")
np.savez("models/norm_params.npz", mean=mean, std=std)