"""
ΕΒΔΟΜΑΔΑ 4: Decision Support (DSS) σε απτά σενάρια.

ΠΕΡΙΕΧΟΜΕΝΟ:
  - Κανόνες κατωφλίου (rule-based alerts) σε δεδομένα αισθητήρων.
  - Πολυκριτηριακή βαθμολόγηση (απλό weighted score).
  - Δημιουργία report (CSV) με ειδοποιήσεις/συστάσεις.
  - Αξιολόγηση απόδοσης με «ψευδές» ground truth (precision/recall).

INPUT:
  Αν υπάρχει 'outputs_week1/cleaned_sensor_data.csv', το χρησιμοποιεί.
  Αλλιώς δημιουργεί συνθετικά δεδομένα.
""" # Περιγραφή αρχείου

from pathlib import Path  # Εισαγωγή Path για ασφαλείς διαδρομές αρχείων
import numpy as np  # NumPy για αριθμητικούς υπολογισμούς και θόρυβο
import pandas as pd  # pandas για DataFrames
try:  # Προσπαθούμε να φορτώσουμε matplotlib
    import matplotlib.pyplot as plt  # Εισαγωγή για γραφήματα
    HAVE_PLOT = True  # Σημαία ότι υπάρχει matplotlib
except Exception:  # Αν αποτύχει η εισαγωγή
    HAVE_PLOT = False  # Δεν κάνουμε plotting αν δεν υπάρχει βιβλιοθήκη

OUT = Path("outputs_week4")  # Φάκελος εξόδου
OUT.mkdir(exist_ok=True, parents=True)  # Δημιουργία φακέλου αν λείπει
W1 = Path("outputs_week1/cleaned_sensor_data.csv")  # Προαιρετικό input από εβδομάδα 1

def unify_columns(df: pd.DataFrame) -> pd.DataFrame:  # Ομογενοποίηση ονομάτων στηλών
    df = df.copy()  # Αντιγραφή για ασφάλεια
    temp_candidates = ["temperature_cal", "temperature", "temp", "t"]  # Πιθανά ονόματα θερμοκρασίας
    hum_candidates = ["humidity_cal", "humidity", "hum", "h"]  # Πιθανά ονόματα υγρασίας
    t_col = next((c for c in temp_candidates if c in df.columns), None)  # Εντοπισμός στήλης θερμοκρασίας
    h_col = next((c for c in hum_candidates if c in df.columns), None)  # Εντοπισμός στήλης υγρασίας
    if t_col is None or h_col is None:  # Αν λείπει κάποια απαιτούμενη στήλη
        raise ValueError("Λείπουν στήλες θερμοκρασίας/υγρασίας στο CSV.")  # Ρίχνουμε σαφές σφάλμα
    df = df.rename(columns={t_col: "temperature_cal", h_col: "humidity_cal"})  # Μετονομασία σε canonical
    if "time" in df.columns:  # Αν υπάρχει στήλη χρόνου
        df["time"] = pd.to_datetime(df["time"], errors="coerce")  # Μετατροπή σε datetime με ανοχή
    else:  # Αν δεν υπάρχει χρόνος
        df["time"] = pd.date_range("2025-01-01", periods=len(df), freq="15min")  # Δημιουργούμε τεχνητό χρόνο
    return df  # Επιστρέφουμε ομογενοποιημένα δεδομένα

def make_synthetic(n=400, seed=5) -> pd.DataFrame:  # Δημιουργία συνθετικών δεδομένων με επεισόδια
    rng = np.random.default_rng(seed)  # Γεννήτρια τυχαίων
    t = pd.date_range("2024-06-01", periods=n, freq="15min")  # Χρονικό index ανά 15 λεπτά
    base_temp = 24 + 4*np.sin(np.linspace(0, 5*np.pi, n))  # Βασική κυματοειδής θερμοκρασία
    base_hum = 50 + 8*np.cos(np.linspace(0, 4*np.pi, n))  # Βασική κυματοειδής υγρασία
    temp = base_temp + rng.normal(0, 0.7, n)  # Προσθήκη θορύβου στη θερμοκρασία
    hum = base_hum + rng.normal(0, 2.0, n)  # Προσθήκη θορύβου στην υγρασία
    hot_slice = slice(n//3, n//3 + 40)  # Παράθυρο επεισοδίου 1
    temp[hot_slice] = 33 + rng.normal(1.5, 0.8, 40)  # Θερμοκρασίες >30°C στο επεισόδιο 1
    hum[hot_slice] = 35 + rng.normal(-2.0, 3.0, 40)  # Υγρασίες <40% στο επεισόδιο 1
    hot_slice2 = slice(n//2, n//2 + 30)  # Παράθυρο επεισοδίου 2
    temp[hot_slice2] = 34 + rng.normal(1.2, 0.7, 30)  # Θερμοκρασίες >30°C στο επεισόδιο 2
    hum[hot_slice2] = 32 + rng.normal(-2.0, 2.5, 30)  # Υγρασίες <40% στο επεισόδιο 2
    df = pd.DataFrame({"time": t, "temperature_cal": temp, "humidity_cal": hum})  # Σύνθεση DataFrame
    return df  # Επιστροφή συνθετικών δεδομένων

def load_data() -> pd.DataFrame:  # Φόρτωση δεδομένων με fallback σε συνθετικά
    if W1.exists():  # Αν το αρχείο της εβδομάδας 1 υπάρχει
        df = pd.read_csv(W1)  # Διαβάζουμε το CSV
        print("✅ Φορτώθηκαν δεδομένα από Week 1.")  # Ενημέρωση χρήστη
        df = unify_columns(df)  # Εναρμόνιση ονομάτων στηλών
    else:  # Αλλιώς δεν έχουμε αρχείο
        print("⚠️ Δεν βρέθηκαν δεδομένα Week 1. Δημιουργώ συνθετικά…")  # Προειδοποίηση
        df = make_synthetic()  # Φτιάχνουμε συνθετικά δεδομένα
    return df  # Επιστροφή DataFrame

def clean_data(df: pd.DataFrame) -> pd.DataFrame:  # Καθαρισμός και προετοιμασία
    df = df.copy()  # Αντιγραφή για να μη μεταβάλλουμε το αρχικό
    df = df.drop_duplicates(subset="time").sort_values("time")  # Αφαίρεση διπλοτύπων και ταξινόμηση χρόνου
    df["temperature_cal"] = pd.to_numeric(df["temperature_cal"], errors="coerce")  # Μετατροπή θερμοκρασίας σε αριθμό
    df["humidity_cal"] = pd.to_numeric(df["humidity_cal"], errors="coerce")  # Μετατροπή υγρασίας σε αριθμό
    df = df.set_index("time")  # Θέτουμε τον χρόνο ως ευρετήριο
    df["temperature_cal"] = df["temperature_cal"].interpolate(method="time").ffill().bfill()  # Παρεμβολή & συμπλήρωση T
    df["humidity_cal"] = df["humidity_cal"].interpolate(method="time").ffill().bfill()  # Παρεμβολή & συμπλήρωση H
    # 🔧 ΔΙΟΡΘΩΣΗ: αντί για dict με tuples στο clip, κάνουμε per-column clip με ξεκάθαρα lower/upper
    df["temperature_cal"] = df["temperature_cal"].clip(lower=-10, upper=60)  # Περικοπή T σε φυσικά όρια
    df["humidity_cal"] = df["humidity_cal"].clip(lower=0, upper=100)  # Περικοπή H σε φυσικά όρια
    return df.reset_index()  # Επαναφορά του time ως κανονικής στήλης

def rule_based_alerts(df: pd.DataFrame, t_thr=30.0, h_thr=40.0):  # Κανόνας κατωφλίου για ειδοποίηση
    df = df.copy()  # Αντιγραφή
    df["alert_drought"] = (df["temperature_cal"] > t_thr) & (df["humidity_cal"] < h_thr)  # Συνθήκη alert
    return df, {"t_thr": t_thr, "h_thr": h_thr}  # Επιστροφή DataFrame και παραμέτρων

def _score_range(x, lo, hi):  # Εσωτερική βαθμολόγηση τιμής ως προς βέλτιστο εύρος
    if pd.isna(x):  # Αν είναι NaN
        return 0.0  # Μηδενική βαθμολογία
    if lo <= x <= hi:  # Αν είναι εντός ορίων
        return 1.0  # Πλήρης βαθμολογία
    if x < lo:  # Αν είναι κάτω από το κάτω όριο
        return max(0.0, 1 - (lo - x) / (max(lo, 1e-6) * 0.5))  # Γραμμική ποινή προς τα κάτω
    return max(0.0, 1 - (x - hi) / (max(hi, 1e-6) * 0.5))  # Γραμμική ποινή προς τα πάνω

def score_row(temp, hum, temp_opt=(18, 28), hum_opt=(45, 70), w_temp=0.6, w_hum=0.4):  # Συνδυαστικός δείκτης
    s_t = _score_range(temp, *temp_opt)  # Βαθμός θερμοκρασίας
    s_h = _score_range(hum, *hum_opt)  # Βαθμός υγρασίας
    return w_temp * s_t + w_hum * s_h  # Σταθμισμένος συνδυασμός

def add_scores(df: pd.DataFrame) -> pd.DataFrame:  # Προσθήκη στήλης score στο DataFrame
    df = df.copy()  # Αντιγραφή
    df["score"] = [score_row(t, h) for t, h in zip(df["temperature_cal"], df["humidity_cal"])]  # Υπολογισμός score
    return df  # Επιστροφή

def fake_ground_truth(df: pd.DataFrame) -> pd.Series:  # Συνθετική «αλήθεια» για αξιολόγηση
    rule_strict = (df["temperature_cal"] > 31) & (df["humidity_cal"] < 38)  # Αυστηρότερος κανόνας
    low_score = df["score"] < 0.25  # Πολύ χαμηλός σύνθετος δείκτης
    y = (rule_strict | low_score).astype(int)  # Ετικέτα 0/1 για ground truth
    return y  # Επιστροφή series

def precision_recall(y_true: pd.Series, y_pred: pd.Series):  # Υπολογισμός μετρικών ταξινόμησης
    tp = int(((y_true == 1) & (y_pred == 1)).sum())  # Αληθείς θετικοί
    fp = int(((y_true == 0) & (y_pred == 1)).sum())  # Ψευδώς θετικοί
    fn = int(((y_true == 1) & (y_pred == 0)).sum())  # Ψευδώς αρνητικοί
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Ακρίβεια (precision)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Ανάκληση (recall)
    return precision, recall, tp, fp, fn  # Επιστροφή όλων των μετρικών

def main():  # Κύρια ροή εκτέλεσης
    df = load_data()  # Φόρτωση/ομογενοποίηση δεδομένων
    df = clean_data(df)  # Καθαρισμός/παρεμβολή/περικοπή
    df, params = rule_based_alerts(df, t_thr=30.0, h_thr=40.0)  # Εφαρμογή κανόνα
    df = add_scores(df)  # Προσθήκη πολυκριτηριακού score
    df["DSS_alert"] = df["alert_drought"] | (df["score"] < 0.3)  # Τελική απόφαση DSS (κανόνας Ή χαμηλό score)
    report_cols = ["time", "temperature_cal", "humidity_cal", "score", "alert_drought", "DSS_alert"]  # Στήλες αναφοράς
    df[report_cols].to_csv(OUT / "dss_report.csv", index=False)  # Αποθήκευση αναφοράς CSV
    print("💾 Αποθηκεύτηκε:", OUT / "dss_report.csv")  # Εκτύπωση διαδρομής αναφοράς
    if HAVE_PLOT:  # Αν υπάρχει matplotlib
        plt.figure()  # Νέο σχήμα
        plt.plot(df["time"], df["score"], label="Score")  # Καμπύλη score στον χρόνο
        plt.axhline(0.3, linestyle="--", label="Όριο συναγερμού (0.3)")  # Γραμμή ορίου
        plt.xlabel("Χρόνος")  # Ετικέτα άξονα Χ
        plt.ylabel("Score [0-1]")  # Ετικέτα άξονα Υ
        plt.title("Πολυκριτηριακός Δείκτης στον χρόνο")  # Τίτλος γραφήματος
        plt.legend()  # Υπόμνημα
        plt.tight_layout()  # Προσαρμογή διατάξεων
        plt.savefig(OUT / "score_time.png", dpi=160)  # Αποθήκευση εικόνας
        print("🖼️", OUT / "score_time.png")  # Ενημέρωση διαδρομής γραφήματος
    else:  # Αν δεν έχουμε matplotlib
        print("ℹ️ Παράλειψη γραφήματος: δεν εντοπίστηκε matplotlib.")  # Μήνυμα ενημέρωσης
    y_true = fake_ground_truth(df)  # Δημιουργία συνθετικής ground truth
    y_pred = df["DSS_alert"].astype(int)  # Μετατροπή απόφασης σε 0/1
    precision, recall, tp, fp, fn = precision_recall(y_true, y_pred)  # Υπολογισμός μετρικών
    with open(OUT / "evaluation.txt", "w", encoding="utf-8") as f:  # Άνοιγμα αρχείου αναφοράς
        f.write(f"Precision={precision:.3f}, Recall={recall:.3f}, TP={tp}, FP={fp}, FN={fn}\n")  # Εγγραφή μετρικών
    print(f"📝 Precision={precision:.3f}, Recall={recall:.3f}, TP={tp}, FP={fp}, FN={fn}")  # Εκτύπωση μετρικών στην κονσόλα

if __name__ == "__main__":  # Εκτέλεση μόνο όταν τρέχει ως κύριο αρχείο
    main()  # Κλήση κύριας ροής
