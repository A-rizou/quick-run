"""
ΕΡΓΑΣΙΑ 1: Ανάλυση & Φιλτράρισμα Δεδομένων Αισθητήρων
ΠΜΣ Ψηφιακές Εφαρμογές και Καινοτομία
-------------------------------------------------------------------
Στόχος: 
- Δημιουργία/Φόρτωση δεδομένων θερμοκρασίας & υγρασίας
- Καθαρισμός & βαθμονόμηση
- Υπολογισμός στατιστικών
- Προσθήκη στηλών με διαφορές από μέσες τιμές (raw & clean)
- Εξαγωγή αποτελεσμάτων και γραφημάτων
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os

OUT = Path("out_w1")
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# 1️⃣ Συνάρτηση δημιουργίας συνθετικών δεδομένων
# ---------------------------------------------------------------------
def make_data(n=400, seed=42):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2024-05-01", periods=n, freq="10min")
    temp = 22 + 5 * np.sin(np.linspace(0, 4 * np.pi, n)) + rng.normal(0, 0.7, n)
    hum = 55 + 8 * np.cos(np.linspace(0, 3 * np.pi, n)) + rng.normal(0, 2.0, n)
    temp[rng.integers(0, n, 6)] = np.nan  # λείπουσες τιμές
    return pd.DataFrame({"time": t, "temperature": temp, "humidity": hum})


# ---------------------------------------------------------------------
# 2️⃣ Φόρτωση υπαρχόντων ή δημιουργία νέων δεδομένων
# ---------------------------------------------------------------------
def load_or_synth(path=None):
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        if "time" not in df.columns:
            raise ValueError("Το CSV πρέπει να περιέχει στήλη 'time'")
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    else:
        print("⚠️ Δεν βρέθηκε CSV· δημιουργώ συνθετικά δεδομένα.")
        df = make_data()
    df = df.dropna(subset=["time"]).sort_values("time")
    return df


# ---------------------------------------------------------------------
# 3️⃣ Καθαρισμός και “βαθμονόμηση” δεδομένων
# ---------------------------------------------------------------------
def clean(df):
    df = df.copy().set_index("time")
    for c in ["temperature", "humidity"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].interpolate(method="time").ffill().bfill()
        lo, hi = df[c].quantile([0.01, 0.99])
        df[c] = df[c].clip(lo, hi)
    df["temp_cal"] = 1.01 * df["temperature"] - 0.2
    df["hum_cal"] = df["humidity"]
    return df.reset_index()


# ---------------------------------------------------------------------
# 4️⃣ Συνάρτηση σχεδίασης γραφημάτων
# ---------------------------------------------------------------------
def plot(df, y, title, fname):
    plt.figure()
    plt.plot(df["time"], df[y], label=y)
    plt.xlabel("Χρόνος")
    plt.ylabel(y)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path = OUT / fname
    plt.savefig(out_path, dpi=150)
    print("🖼️ Αποθηκεύτηκε:", out_path)


# ---------------------------------------------------------------------
# 5️⃣ Υπολογισμός βασικών στατιστικών
# ---------------------------------------------------------------------
def compute_stats(df, cols):
    stats = df[cols].describe().T
    stats["variance"] = df[cols].var()
    stats.rename(
        columns={
            "mean": "Μέσος Όρος",
            "std": "Τυπική Απόκλιση",
            "min": "Ελάχιστο",
            "max": "Μέγιστο",
        },
        inplace=True,
    )
    return stats[["Μέσος Όρος", "Τυπική Απόκλιση", "Ελάχιστο", "Μέγιστο", "variance"]]


# ---------------------------------------------------------------------
# 6️⃣ Αποθήκευση σε CSV/JSON
# ---------------------------------------------------------------------
def save_csv_and_json(df, name):
    csv_path = OUT / f"{name}.csv"
    json_path = OUT / f"{name}.json"
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", date_format="iso", force_ascii=False, indent=2)
    print(f"💾 Αποθηκεύτηκε: {csv_path}, {json_path}")


# ---------------------------------------------------------------------
# 7️⃣ Κύριο πρόγραμμα
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Διαδρομή σε CSV (προαιρετικό)")
    args = parser.parse_args()

    # Φόρτωση ή δημιουργία δεδομένων
    df_raw = load_or_synth(args.csv)
    save_csv_and_json(df_raw, "raw_data")

    # Καθαρισμός
    df_clean = clean(df_raw)
    save_csv_and_json(df_clean, "clean_data")

    # --- Υπολογισμός μέσων τιμών ---
    mean_temp_raw = df_raw["temperature"].mean()
    mean_hum_raw = df_raw["humidity"].mean()
    mean_temp_clean = df_clean["temp_cal"].mean()
    mean_hum_clean = df_clean["hum_cal"].mean()

    # --- Προσθήκη στηλών διαφορών από τη μέση τιμή ---
    df_raw["temp_diff_raw"] = df_raw["temperature"] - mean_temp_raw
    df_raw["hum_diff_raw"] = df_raw["humidity"] - mean_hum_raw

    df_clean["temp_diff_clean"] = df_clean["temp_cal"] - mean_temp_clean
    df_clean["hum_diff_clean"] = df_clean["hum_cal"] - mean_hum_clean

    # Αποθήκευση νέων DataFrames
    save_csv_and_json(df_raw, "raw_with_differences")
    save_csv_and_json(df_clean, "clean_with_differences")

    # --- Σχεδίαση γραφημάτων ---
    plot(df_raw, "temperature", "Θερμοκρασία (Raw)", "temp_raw.png")
    plot(df_clean, "temp_cal", "Θερμοκρασία (Καθαρή)", "temp_clean.png")

    # --- Στατιστικά σύγκρισης ---
    stats_raw = compute_stats(df_raw, ["temperature", "humidity"])
    stats_clean = compute_stats(df_clean, ["temp_cal", "hum_cal"])

    compare = pd.concat(
        [stats_raw.add_prefix("Raw_"), stats_clean.add_prefix("Clean_")], axis=1
    )

    save_csv_and_json(compare.reset_index().rename(columns={"index": "Μεταβλητή"}), "statistics_comparison")

    print("\n📊 Στατιστικά σύγκρισης (Raw vs Clean):\n")
    print(compare.round(3))

    print("\n📈 Μέσες τιμές:")
    print(f"Raw: Θερμοκρασία={mean_temp_raw:.2f}, Υγρασία={mean_hum_raw:.2f}")
    print(f"Clean: Θερμοκρασία={mean_temp_clean:.2f}, Υγρασία={mean_hum_clean:.2f}")


if __name__ == "__main__":
    main()
