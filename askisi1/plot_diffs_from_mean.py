import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

out = Path("out_w1")

# Διαβάζουμε τα δεδομένα που φτιάξαμε πριν
df = pd.read_csv(out / "diff_from_mean_all.csv", parse_dates=["time"])

# --- Temperature ---
plt.figure()
plt.plot(df["time"], df["temp_diff_raw_mean"], label="Temp (raw - mean)")
plt.plot(df["time"], df["temp_diff_clean_mean"], label="Temp (clean - mean)")
plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
plt.xlabel("Χρόνος")
plt.ylabel("Διαφορά από μέση θερμοκρασία (°C)")
plt.title("Διαφορά θερμοκρασίας από μέση τιμή (raw vs clean)")
plt.legend()
plt.tight_layout()
plt.savefig(out / "temp_diff_from_mean.png", dpi=150)
plt.close()

# --- Humidity ---
plt.figure()
plt.plot(df["time"], df["hum_diff_raw_mean"], label="Hum (raw - mean)")
plt.plot(df["time"], df["hum_diff_clean_mean"], label="Hum (clean - mean)")
plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
plt.xlabel("Χρόνος")
plt.ylabel("Διαφορά από μέση υγρασία (%)")
plt.title("Διαφορά υγρασίας από μέση τιμή (raw vs clean)")
plt.legend()
plt.tight_layout()
plt.savefig(out / "hum_diff_from_mean.png", dpi=150)
plt.close()

print("✅ Δημιουργήθηκαν: temp_diff_from_mean.png, hum_diff_from_mean.png")
