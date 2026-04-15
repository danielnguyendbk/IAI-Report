import pandas as pd
import numpy as np

# Đọc raw file từ UCI
df = pd.read_csv("raw_data/arrhythmia.data", header=None)

# UCI dùng "?" cho missing values
df = df.replace("?", np.nan)

# Chuyển mọi cột về numeric nếu có thể
df = df.apply(pd.to_numeric, errors="coerce")

# Điền missing bằng median từng cột
df = df.fillna(df.median(numeric_only=True))

# Xuất ra CSV sạch
df.to_csv("data/Arrhythmia_raw_clean.csv", index=False, header=False)

print(df.shape)