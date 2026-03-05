import pandas as pd
import numpy as np

def load_t200_pwm_cur(src: str, sheet: str):
    "Loads the PWM and current data from the T200 performance data Excel file. Returns two numpy arrays: PWM values and corresponding current values."
    
    df = pd.read_excel(src, sheet_name = sheet)

    PWM_Col = df[(df.columns[0])].to_numpy()
    CUR_Col = df[(df.columns[2])].to_numpy()

    mask = np.isfinite(PWM_Col) & np.isfinite(CUR_Col)
    PWM_Col = PWM_Col[mask]
    CUR_Col = CUR_Col[mask]

    idx = np.argsort(PWM_Col)
    PWM_Col = PWM_Col[idx]
    CUR_Col = CUR_Col[idx]

    return PWM_Col, CUR_Col
    