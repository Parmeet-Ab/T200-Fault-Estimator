import numpy as np 
import tensorflow as tf
from tensorflow import keras
from simulate_data import generate_pwm, pwm_to_current, add_spike
import matplotlib.pyplot as plt


Window = 20

model = keras.models.load_model("models/t200_autoencoder.keras")
rng = np.random.default_rng()

#Make new signal to test model on 
pwm = generate_pwm(200, rng)
current = pwm_to_current(pwm, '12 V')
current = current + rng.normal(0, 0.15, size=current.shape) #simulates sensor noise

#inject a fault so we can test detection 
current_fault = add_spike(current, rng)

data = np.load("models/norm_params.npz")
mean = data["mean"]
std = data["std"]
current_fault_norm = (current_fault - mean) / std
current_fault_norm = current_fault_norm.reshape(1, -1)

recon = model.predict(current_fault_norm, verbose=0)[0]

err_t = (current_fault_norm[0] - recon) ** 2

scores = []
for i in range(200 - Window + 1):
    scores.append(np.mean(err_t[i:i+Window]))
scores = np.array(scores)


#Creating and Formatting Graphs
fig, ax1 = plt.subplots()

ax1.plot(current_fault, label="Faulty Current", color="orange")
ax1.set_xlabel("Time step")
ax1.set_ylabel("Current (A)", color="orange")
ax1.tick_params(axis='y', labelcolor="orange")

ax2 = ax1.twinx()
ax2.plot(range(Window - 1, len(scores) + Window - 1), scores, label="Anomaly Score", color="red", linestyle="--") 
ax2.set_ylabel("Anomaly Score", color="red")
ax2.tick_params(axis='y', labelcolor="red")

plt.title("Fault Detection on Thruster Current Signal")
plt.savefig("images/detection_example_plot.png")
plt.show()
