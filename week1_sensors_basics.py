# ΕΒΔ.1 (με σχόλια σε κάθε γραμμή): Φόρτωση/σύνθεση δεδομένων αισθητήρων, καθαρισμός, «βαθμονόμηση», γραφήματα.  # περιγραφή αρχείου

import argparse                                                                         # βιβλιοθήκη για διάβασμα ορισμάτων γραμμής εντολών
import os                                                                               # βοηθητικές συναρτήσεις λειτουργικού (π.χ. έλεγχος αν υπάρχει αρχείο)
from pathlib import Path                                                                # ασφαλείς διαδρομές αρχείων/φακέλων ανεξαρτήτως λειτουργικού
import numpy as np                                                                      # αριθμητικοί υπολογισμοί και τυχαίοι αριθμοί
import pandas as pd                                                                     # διαχείριση πινάκων/δεδομένων (DataFrame)
import matplotlib.pyplot as plt                                                         # σχεδίαση γραφημάτων
from pathlib import Path


OUT = Path("out_w1")                                                                    # όνομα φακέλου εξόδων για αποθήκευση αποτελεσμάτων
OUT.mkdir(exist_ok=True)                                                                # δημιουργία φακέλου αν δεν υπάρχει ήδη

def save_csv_and_json(df, csv_path, orient="records", lines=False):
    """
    Αποθηκεύει df σε CSV ΚΑΙ JSON σε ένα βήμα.
    - orient="records": λίστα από αντικείμενα {col: value}
    - date_format="iso": ISO-8601 για ημερομηνίες
    - lines=True: NDJSON (1 γραμμή/εγγραφή) για πολύ μεγάλα αρχεία
    """
    csv_path = Path(csv_path)
    df.to_csv(csv_path, index=False)
    json_path = csv_path.with_suffix(".json")
    df.to_json(json_path, orient=orient, date_format="iso", force_ascii=False, indent=2, lines=lines)
    print(f"💾 CSV:  {csv_path}")
    print(f"💾 JSON: {json_path}")
    

def make_data(n=400, seed=42):                                                             # συνάρτηση: δημιουργεί συνθετικά δεδομένα αν δεν δώσουμε CSV
    rng = np.random.default_rng(seed)                                                      # αρχικοποίηση γεννήτριας τυχαιοτήτων για αναπαραγωγιμότητα
    t = pd.date_range("2024-05-01", periods=n, freq="10min")                               # χρονικός άξονας: n δείγματα ανά 10 λεπτά
    temp = 22 + 5*np.sin(np.linspace(0, 4*np.pi, n)) + rng.normal(0, 0.7, n)               # θερμοκρασία: ημερήσια κυματομορφή + θόρυβος
    hum  = 55 + 8*np.cos(np.linspace(0, 3*np.pi, n)) + rng.normal(0, 2.0, n)               # υγρασία: κυματομορφή + θόρυβος
    temp[rng.integers(0, n, 6)] = np.nan                                                   # εισαγωγή μερικών λειπουσών τιμών για εξάσκηση καθαρισμού
    return pd.DataFrame({"time": t, "temperature": temp, "humidity": hum})                 # επιστροφή ως DataFrame με τρεις στήλες

def load_or_synth(path):                                                                   # συνάρτηση: φόρτωση CSV ή δημιουργία συνθετικών
    if path and os.path.exists(path):                                                      # αν δόθηκε διαδρομή και το αρχείο υπάρχει
        df = pd.read_csv(path)                                                             # φορτώνουμε το CSV σε DataFrame
        if "time" not in df.columns:                                                       # ελέγχουμε ότι υπάρχει στήλη time
            raise ValueError("Το CSV πρέπει να έχει στήλη 'time'.")                        # μήνυμα σφάλματος αν λείπει
        df["time"] = pd.to_datetime(df["time"], errors="coerce")                           # μετατροπή στήλης time σε τύπο datetime (μη έγκυρα -> NaT)
    else:                                                                                  # αλλιώς (δεν δόθηκε ή δεν βρέθηκε CSV)
        print("⚠️ Δεν βρέθηκε CSV· δημιουργώ συνθετικά.")                                  # ενημερωτικό μήνυμα προς τον χρήστη
        df = make_data()                                                                   # δημιουργούμε συνθετικά δεδομένα
    df = df.dropna(subset=["time"]).sort_values("time")                                    # αφαιρούμε γραμμές με άκυρο χρόνο και ταξινομούμε χρονικά
    return df                                                                              # επιστρέφουμε τον «ακατέργαστο» πίνακα

def clean(df):                                                                             # συνάρτηση: καθαρισμός/προεπεξεργασία δεδομένων
    df = df.copy()                                                                         # αντιγραφή για να μη μεταβάλλουμε το αρχικό αντικείμενο
    df = df.set_index("time")                                                              # ορισμός της στήλης time ως ευρετήριο (DatetimeIndex)
    for c in ["temperature", "humidity"]:                                                  # για κάθε μετρούμενη στήλη
        df[c] = pd.to_numeric(df[c], errors="coerce")                                      # μετατροπή σε αριθμητικό τύπο (μη αριθμητικά -> NaN)
        df[c] = df[c].interpolate(method="time").ffill().bfill()                           # παρεμβολή στον χρόνο + συμπλήρωση άκρων (μπρος/πίσω)
        lo, hi = df[c].quantile([0.01, 0.99])                                              # υπολογισμός 1ου και 99ου εκατοστημορίου (για outliers)
        df[c] = df[c].clip(lo, hi)                                                         # περικοπή τιμών εκτός εύρους (winsorization)
    df["temp_cal"] = 1.01*df["temperature"] - 0.2                                          # απλή «βαθμονόμηση» θερμοκρασίας: gain & offset (ενδεικτικά)
    df["hum_cal"]  = df["humidity"]                                                        # για υγρασία κρατάμε ως έχει (ή θα μπαίνει ανάλογη διόρθωση)
    return df.reset_index()                                                                # επαναφορά της στήλης time (όχι πια ως index) και επιστροφή

def plot(df, y, title, fname):                                                             # συνάρτηση: δημιουργεί και αποθηκεύει ένα γράφημα γραμμής
    plt.figure()                                                                           # νέο σχήμα matplotlib
    plt.plot(df["time"], df[y], label=y)                                                   # καμπύλη: y ως συνάρτηση του χρόνου
    plt.xlabel("Χρόνος")                                                                   # ετικέτα άξονα x
    plt.ylabel(y)                                                                          # ετικέτα άξονα y
    plt.title(title)                                                                       # τίτλος διαγράμματος
    plt.legend()                                                                           # εμφάνιση υπομνήματος
    plt.tight_layout()                                                                     # συμπίεση κενών περιθωρίων για καθαρή εμφάνιση
    p = OUT / fname                                                                        # πλήρης διαδρομή αποθήκευσης εικόνας
    plt.savefig(p, dpi=150)                                                                # αποθήκευση εικόνας με ανάλυση 150 dpi
    print("🖼️", p)                                                                         # εκτύπωση τοποθεσίας αρχείου εικόνας

def main():                                                                                # κύρια ροή εκτέλεσης προγράμματος
    ap = argparse.ArgumentParser()                                                         # δημιουργία parser για ορίσματα γραμμής εντολών
    ap.add_argument("--csv", type=str, default=None, help="CSV με στήλες time,temperature,humidity")  # ορισμός ορίσματος --csv
    args = ap.parse_args()                                                                 # ανάγνωση ορισμάτων από τη γραμμή εντολών

    df_raw = load_or_synth(args.csv)                                                       # φόρτωση πραγματικών ή δημιουργία συνθετικών δεδομένων
    
    out_csv = OUT / "raw.csv"                                                              # διαδρομή αρχείου για αποθήκευση μη καθαρισμένων δεδομένων
    save_csv_and_json(df_raw, out_csv)                                                     # αποθήκευση DataFrame σε CSV χωρίς ευρετήριο & json 
    
    print("💾 Αποθηκεύτηκε:", out_csv)      
    
    plot(df_raw, "temperature", "Θερμοκρασία", "temp_raw.png")                             # γράφημα για μη καθαρισμένη θερμοκρασία
    plot(df_raw, "humidity",  "Υγρασία", "hum_raw.png")                                    # γράφημα για μη καθαρισμένη υγρασία
    
    df = clean(df_raw)                                                                     # καθαρισμός/επεξεργασία δεδομένων

    out_csv = OUT / "clean.csv"                                                            # διαδρομή αρχείου για αποθήκευση καθαρισμένων δεδομένων
    save_csv_and_json(df, out_csv)                                                         # αποθήκευση DataFrame σε CSV χωρίς ευρετήριο & json 
    
    print("💾 Αποθηκεύτηκε:", out_csv)                                                     # ενημέρωση ότι αποθηκεύτηκε

    plot(df, "temp_cal", "Θερμοκρασία (βαθμονομημένη)", "temp_cal.png")                    # γράφημα για βαθμονομημένη θερμοκρασία
    plot(df, "hum_cal",  "Υγρασία (βαθμονομημένη)",     "hum_cal.png")                     # γράφημα για βαθμονομημένη υγρασία

    
    ############################

    # --- Ευθυγράμμιση raw vs clean και υπολογισμός διαφορών ---

    # 1) Κρατάμε μόνο ό,τι μας χρειάζεται από το raw και μετονομάζουμε για σαφήνεια
    raw_sel = (
        df_raw[["time", "temperature", "humidity"]]
        .copy()
        .rename(columns={"temperature": "temp_raw", "humidity": "hum_raw"})
    )

    # 2) Ευθυγράμμιση με το καθαρισμένο df πάνω στο time (inner join = μόνο κοινές χρονικές στιγμές)
    aligned = (
        raw_sel
        .merge(df[["time", "temp_cal", "hum_cal"]], on="time", how="inner")
        .sort_values("time")
    )

    # 3) Νέο DataFrame με διαφορές (από clean − raw) και % διαφορές

    df_diff = aligned.assign(
        temp_delta = aligned["temp_cal"] - aligned["temp_raw"],
        hum_delta  = aligned["hum_cal"]  - aligned["hum_raw"],
        temp_delta_pct = 100 * (aligned["temp_cal"] - aligned["temp_raw"]) / aligned["temp_raw"].replace(0, np.nan),
        hum_delta_pct  = 100 * (aligned["hum_cal"]  - aligned["hum_raw"])  / aligned["hum_raw"].replace(0, np.nan)
    )

    # 4) (προαιρετικό) Αποθήκευση για να το δείτε/μοιραστείτε
    diff_path = OUT / "diff_raw_vs_clean.csv"
    save_csv_and_json(df_diff, diff_path)                                                # αποθήκευση DataFrame σε CSV χωρίς ευρετήριο & json 
    
    print("📄 Διαφορές raw vs clean αποθηκεύτηκαν:", diff_path)

    plot(df_diff, "temp_delta_pct", "% Δ Θερμοκρασίας", "temp_delta.png")                # γράφημα για Δ θερμοκρασίας
    plot(df_diff, "hum_delta_pct",  "% Δ Υγρασίας",     "hum_delta.png")                 # γράφημα για Δ υγρασίας

    ############################


    print(df[["temp_cal","hum_cal"]].describe())                                         # συνοπτικά στατιστικά περιγραφής στην κονσόλα

if __name__ == "__main__":                                                               # έλεγχος: αν το αρχείο τρέχει ως κύριο πρόγραμμα (όχι ως import)
    main()                                                                               # κάλεσε την main() για να εκτελεστεί η ροή