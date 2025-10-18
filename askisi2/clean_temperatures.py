import re
import pandas as pd
from pathlib import Path

# === Ρυθμίσεις αρχείων ===
IN = Path("random_temperatures.csv")   # πηγαίο αρχείο (αν έχει άλλο όνομα, άλλαξέ το)
OUT = Path("random_temperatures_clean.csv")

# === Καθαρισμός αριθμών ===
SEP_PATTERN = r"[.,·•‧˙∙\u00B7\u2024\u2027\u2219\u22C5\u30FB\uFF0E\s\u00A0]+"

def clean_weird_number(s):
    """Μετατρέπει '25.914.151.239.263.200' -> 25.914151239263200 (float)"""
    if pd.isna(s):
        return float('nan')
    s = str(s).strip()
    sign = ""
    if s.startswith("-"):
        sign, s = "-", s[1:]
    elif s.startswith("+"):
        s = s[1:]
    parts = re.split(SEP_PATTERN, s)
    parts = [p for p in parts if p != ""]
    if not parts:
        return float('nan')
    if len(parts) == 1:
        return float(sign + parts[0])
    int_part = parts[0]
    frac_part = "".join(parts[1:])
    normalized = f"{sign}{int_part}.{frac_part}"
    try:
        return float(normalized)
    except ValueError:
        digits = re.sub(r"\D", "", sign + int_part + frac_part)
        if not digits:
            return float('nan')
        k = len(int_part)
        normalized = f"{sign}{digits[:k]}.{digits[k:]}"
        return float(normalized)

# === Κύρια ροή ===
def main():
    df = pd.read_csv(IN)
    col = "temperature"
    if col not in df.columns:
        raise SystemExit(f"❌ Δεν βρέθηκε στήλη '{col}' στο {IN}")
    
    # Καθαρισμός και αποθήκευση
    df["temperature_clean"] = df[col].apply(clean_weird_number)
    df[["temperature_clean"]].to_csv(OUT, index=False)
    print(f"✔ Καθαρό αρχείο έτοιμο: {OUT.resolve()}")

    # === Υπολογισμός στατιστικών ===
    mean = df["temperature_clean"].mean()
    median = df["temperature_clean"].median()
    min_val = df["temperature_clean"].min()
    max_val = df["temperature_clean"].max()
    var = df["temperature_clean"].var()
    std = df["temperature_clean"].std()

    print("\n📊 Βασικά Στατιστικά (από καθαρισμένα δεδομένα)")
    print(f"Μέσος Όρος: {mean}")
    print(f"Διάμεσος: {median}")
    print(f"Ελάχιστο: {min_val}")
    print(f"Μέγιστο: {max_val}")
    print(f"Διακύμανση: {var}")
    print(f"Τυπική Απόκλιση: {std}")

if __name__ == "__main__":
    main()

