import pandas as pd
from pathlib import Path

out = Path("out_w1")
raw = pd.read_csv(out/"raw.csv", parse_dates=["time"])
clean = pd.read_csv(out/"clean.csv", parse_dates=["time"])

# Μέσοι όροι (αγνοούν NaN)
m_temp_raw = raw["temperature"].mean(skipna=True)
m_hum_raw  = raw["humidity"].mean(skipna=True)
m_temp_cln = clean["temp_cal"].mean(skipna=True)
m_hum_cln  = clean["hum_cal"].mean(skipna=True)

# Διαφορές από μέση τιμή (signed)
raw["temp_diff_raw_mean"] = raw["temperature"] - m_temp_raw
raw["hum_diff_raw_mean"]  = raw["humidity"]    - m_hum_raw
clean["temp_diff_clean_mean"] = clean["temp_cal"] - m_temp_cln
clean["hum_diff_clean_mean"]  = clean["hum_cal"]  - m_hum_cln

# Κρατάμε μόνο ό,τι χρειαζόμαστε και ευθυγραμμίζουμε στον χρόνο
raw_sel   = raw[["time","temp_diff_raw_mean","hum_diff_raw_mean"]]
clean_sel = clean[["time","temp_diff_clean_mean","hum_diff_clean_mean"]]

df = raw_sel.merge(clean_sel, on="time", how="inner").sort_values("time")

# Αποθήκευση
csv_path = out/"diff_from_mean_all.csv"
df.to_csv(csv_path, index=False)
df.to_json(out/"diff_from_mean_all.json", orient="records", date_format="iso", force_ascii=False, indent=2)

print(f"OK -> {csv_path}")
