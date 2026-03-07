import numpy as np 
from simulate_data import generate_pwm, pwm_to_current, add_spike

sheet_name = '12 V'

def main():
    rng = np.random.default_rng(0)  # Set a fixed seed for reproducibility
    
    N_Norm = 2000 
    N_Fault = 500
    Seq_Len = 200

    X_normal = np.zeros((N_Norm, Seq_Len), dtype=np.float32)
    X_fault = np.zeros((N_Fault, Seq_Len), dtype=np.float32)

    for i in range(N_Norm):
        "Generates synthetic normal current data based on the T200 performance curve, with added sensor noise. Stores the data in the X_normal array."

        pwm = generate_pwm(Seq_Len, rng)
        cur = pwm_to_current(pwm, sheet_name)
        cur = cur + rng.normal(0, 0.15, size=Seq_Len) #simulates sensor noise
        X_normal[i] = cur.astype(np.float32)

    for i in range(N_Fault):
        "Generates synthetic fault current data by first creating normal current data and then adding a random spike to simulate a fault condition. Stores the data in the X_fault array."

        pwm = generate_pwm(Seq_Len, rng)
        cur = pwm_to_current(pwm)
        cur = cur + rng.normal(0, 0.15, size=Seq_Len)
        X_fault[i] = add_spike(cur, rng).astype(np.float32)
    
    np.savez_compressed("data/t200_dataset.npz", X_normal=X_normal, X_fault=X_fault)

if __name__ == "__main__":
    main()
    