import pandas as pd
import numpy as np

df = pd.read_excel('data/T200-Public-Performance-Data-10-20V-September-2019.xlsx', sheet_name = '12 V')

PWM_Col = df[(df.columns[0])].to_numpy()
CUR_Col = df[(df.columns[2])].to_numpy()

mask = np.isfinite(PWM_Col) & np.isfinite(CUR_Col)
PWM_Col = PWM_Col[mask]
CUR_Col = CUR_Col[mask]

idx = np.argsort(PWM_Col)
PWM_Col = PWM_Col[idx]
CUR_Col = CUR_Col[idx]

est_current = np.interp(1700, PWM_Col, CUR_Col)
print(est_current)