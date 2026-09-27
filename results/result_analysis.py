import numpy as np
import pandas as pd
import os

pred = np.load('only-ml/pred.npy')
true = np.load('only-ml/true.npy')
print("hi")
abs_error = np.abs(pred - true)
mae_per_column = np.mean(abs_error, axis=0)
max_ae_per_column = np.max(abs_error, axis=0)

# Print results with labels
column_names = ['Yaw Rate', 'Vx', 'Vy', 'Ax', 'Ay']
print("Error Metrics (MAE and Max AE):\n")
for name, mae, maxae in zip(column_names, mae_per_column, max_ae_per_column):
    print(f"{name:>8} | MAE: {mae:.6f} | Max AE: {maxae:.6f}")

column_names = ['Yaw Rate', 'Vx', 'Vy', 'Ax', 'Ay']

print("Column-wise statistics for 'pred':\n")
for i, name in enumerate(column_names):
    col = pred[:, i]
    print(f"{name:>8}:")
    print(f"  Mean     : {np.mean(col):.6f}")
    print(f"  Std Dev  : {np.std(col):.6f}")
    print(f"  Min      : {np.min(col):.6f}")
    print(f"  Max      : {np.max(col):.6f}")
    print(f"  Median   : {np.median(col):.6f}")
    print(f"  5th pct  : {np.percentile(col, 5):.6f}")
    print(f"  95th pct : {np.percentile(col, 95):.6f}")
    print()