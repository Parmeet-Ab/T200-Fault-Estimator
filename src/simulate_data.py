import numpy as np 
import matplotlib.pyplot as plt
from load_data import load_t200_pwm_cur

file_path = 'data/T200-Public-Performance-Data-10-20V-September-2019.xlsx'
sheet_name = '12 V'
pwm_curve, cur_curve = load_t200_pwm_cur(file_path, sheet_name)

def generate_pwm(N, rng=None):
    "Generates synthetic PWM and current data based on the T200 performance curve. Returns two numpy arrays: PWM values and corresponding current values."
    if rng is None:
        rng = np.random.default_rng()
    t = np.arange(N) #time steps

    amp=rng.uniform(150, 350) #creates random amplitude for the PWM signal between 150 and 350
    pwm_t = 1500 + amp*np.sin(t/20) #changes pwm data to be based on time
    
    return np.clip(pwm_t, 1100, 1900)

def pwm_to_current(pwm_t):
    "Converts PWM values to current values using interpolation based on the T200 performance curve."
    return np.interp(pwm_t, pwm_curve, cur_curve)

def add_spike(current, rng = None):
    "Adds a random spike to the current data to simulate a fault condition. Returns the modified current array."
    if rng is None:
        rng = np.random.default_rng()
    current_fault = current.copy()
    spike_index = rng.integers(20, len(current_fault)-20) #chooses a random index to add the spike, avoiding the edges
    current_fault[spike_index] += rng.uniform(4, 10) #adds a random spike between 5 and 10 amps
    return current_fault


def demo():
    N = 200
    pwm_t = generate_pwm(N)
    current_t = pwm_to_current(pwm_t)
    current_fault = add_spike(current_t)

    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Time step")
    ax1.set_ylabel("PWM (µs)", color="blue")
    ax1.plot(pwm_t, color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Current (A)")
    ax2.plot(current_t, color="orange", label="Current")
    ax2.plot(current_fault, color="red", linestyle="--", label="Fault")
    plt.title("Thruster PWM → Current Simulation")
    plt.show()

if __name__ == "__main__":
    demo()