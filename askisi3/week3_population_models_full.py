# week3_population_models_full.py  # όνομα αρχείου για αναφορά
# Σκοπός: Μοντέλα πληθυσμών (Λογιστικό & Lotka–Volterra) με οπτικοποίηση και ευαισθησία  # περιγραφή
# Εκτελεί: 1) λογιστικό διακριτό με θόρυβο, 2) Lotka–Volterra με RK4, 3) sweep παραμέτρων + heatmap  # σύνοψη

from pathlib import Path  # διαχείριση διαδρομών
import numpy as np        # αριθμητικοί υπολογισμοί
import pandas as pd       # αποθήκευση αποτελεσμάτων σε CSV
import matplotlib.pyplot as plt  # γραφήματα

OUT = Path("out_w3"); OUT.mkdir(exist_ok=True)  # δημιουργία φακέλου εξόδων

# ---------------- ΛΟΓΙΣΤΙΚΟ ΜΟΝΤΕΛΟ (ΔΙΑΚΡΙΤΟ) ----------------  # ενότητα
def logistic_discrete(P0=15, r=0.28, K=900, steps=160, noise_std=2.5, seed=7):  # ορισμός συνάρτησης με προεπιλογές
    rng = np.random.default_rng(seed)  # τυχαίος γεννήτορας
    P = np.empty(steps)                # προ-δέσμευση πίνακα πληθυσμού
    P[0] = float(P0)                   # αρχικός πληθυσμός
    for t in range(1, steps):          # επανάληψη σε βήματα χρόνου
        growth = r * P[t-1] * (1 - P[t-1] / K)  # λογιστική αύξηση
        noise = rng.normal(0, noise_std)        # θόρυβος κανονικός
        P[t] = max(0.0, P[t-1] + growth + noise)  # ενημέρωση με μη αρνητικό κατώφλι
    return P                           # επιστροφή τροχιάς πληθυσμού

def plot_series(x, y, title, fname, xlabel="Χρόνος (βήματα)", ylabel="Πληθυσμός"):  # βοηθητικό για γραφήματα
    plt.figure()                       # νέα φιγούρα
    plt.plot(x, y, marker="o", markersize=3, linewidth=1, label="P")  # καμπύλη με δείκτες
    plt.xlabel(xlabel)                 # ετικέτα άξονα x
    plt.ylabel(ylabel)                 # ετικέτα άξονα y
    plt.title(title)                   # τίτλος
    plt.legend()                       # υπόμνημα
    plt.tight_layout()                 # τακτοποίηση περιθωρίων
    p = OUT / fname                    # πλήρης διαδρομή αρχείου εικόνας
    plt.savefig(p, dpi=160)            # αποθήκευση PNG
    print("🖼️", p)                    # μήνυμα τοποθεσίας γραφήματος

# ---------------- ODE ΟΛΟΚΛΗΡΩΣΗ (RK4 ΓΕΝΙΚΟΣ) ----------------  # ενότητα
def rk4(f, y0, h, n):                 # κλασικός ολοκληρωτής 4ης τάξης
    y = np.zeros((n + 1, len(y0)))    # πίνακας λύσης
    y[0] = y0                         # αρχική συνθήκη
    for i in range(n):                # βήματα ολοκλήρωσης
        k1 = f(y[i])                  # κλίση 1
        k2 = f(y[i] + h * k1 / 2)     # κλίση 2
        k3 = f(y[i] + h * k2 / 2)     # κλίση 3
        k4 = f(y[i] + h * k3)         # κλίση 4
        y[i + 1] = y[i] + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)  # ενημέρωση κατά RK4
    return y                          # επιστροφή όλων των σημείων

# ---------------- LOTKA–VOLTERRA (ΘΗΡΕΥΤΗΣ–ΘΗΡΑΜΑ) ----------------  # ενότητα
def lotka_volterra(alpha=0.55, beta=0.028, delta=0.012, gamma=0.45,  # παράμετροι συστήματος
                   x0=35, y0=10, T=120, h=0.1):                      # αρχικές & χρονισμός
    def f(v):                                   # δεξί μέλος ODE
        x, y = v                                # αποσυσκευασία θήραμα/θηρευτής
        dx = alpha * x - beta * x * y           # εξίσωση θηράματος
        dy = delta * x * y - gamma * y          # εξίσωση θηρευτή
        return np.array([dx, dy], float)        # επιστροφή διανύσματος
    n = int(T / h)                              # αριθμός βημάτων
    sol = rk4(f, np.array([x0, y0], float), h, n)  # ολοκλήρωση με RK4
    t = np.arange(n + 1) * h                    # διάνυσμα χρόνου
    prey = sol[:, 0]                            # απομόνωση θηράματος
    pred = sol[:, 1]                            # απομόνωση θηρευτή
    return t, prey, pred                        # επιστροφή χρονικής σειράς και πληθυσμών

def plot_lv_time(t, prey, pred, fname):         # χρονική απεικόνιση LV
    plt.figure()                                # νέα φιγούρα
    plt.plot(t, prey, label="Θήραμα")           # καμπύλη θηράματος
    plt.plot(t, pred, label="Θηρευτής")         # καμπύλη θηρευτή
    plt.xlabel("Χρόνος")                        # ετικέτα x
    plt.ylabel("Πληθυσμός")                     # ετικέτα y
    plt.title("Lotka–Volterra: Εξέλιξη στο χρόνο")  # τίτλος
    plt.legend()                                # υπόμνημα
    plt.tight_layout()                          # περιθώρια
    p = OUT / fname                             # έξοδος αρχείου
    plt.savefig(p, dpi=160)                     # αποθήκευση
    print("🖼️", p)                              # ενημέρωση

def plot_lv_phase(prey, pred, fname):              # φασικό διάγραμμα LV
    plt.figure()                                   # νέα φιγούρα
    plt.plot(prey, pred)                           # τροχιά στο επίπεδο (x,y)
    plt.xlabel("Θήραμα (x)")                       # ετικέτα x
    plt.ylabel("Θηρευτής (y)")                     # ετικέτα y
    plt.title("Lotka–Volterra: Φασικό Διάγραμμα")  # τίτλος
    plt.tight_layout()                             # περιθώρια
    p = OUT / fname                                # έξοδος αρχείου
    plt.savefig(p, dpi=160)                        # αποθήκευση
    print("🖼️", p)                                 # ενημέρωση

# ---------------- ΕΥΑΙΣΘΗΣΙΑ / ΣΑΡΩΣΗ ΠΑΡΑΜΕΤΡΩΝ ----------------  # ενότητα
def logistic_peak(P0, r, K, steps=140, noise_std=0.0):  # βοηθητικό για μετρική κορυφής
    P = logistic_discrete(P0=P0, r=r, K=K, steps=steps, noise_std=noise_std, seed=123)  # προσομοίωση
    return np.max(P)                                # επιστροφή μέγιστου πληθυσμού

def sweep_logistic(P0_vals, r_vals, K=900):         # σάρωση σε πλέγμα τιμών
    heat = np.zeros((len(P0_vals), len(r_vals)))    # πίνακας heatmap
    for i, P0 in enumerate(P0_vals):                # βρόχος σε αρχικά
        for j, r in enumerate(r_vals):              # βρόχος σε ρυθμούς ανάπτυξης
            heat[i, j] = logistic_peak(P0, r, K)    # υπολογισμός κορυφής
    return heat                                     # επιστροφή πλέγματος

def plot_heatmap(P0_vals, r_vals, Z, fname):        # σχεδίαση heatmap
    plt.figure()                                    # νέα φιγούρα
    extent = [min(r_vals), max(r_vals), min(P0_vals), max(P0_vals)]  # όρια αξόνων
    plt.imshow(Z, aspect="auto", origin="lower", extent=extent)      # απεικόνιση πλέγματος
    plt.colorbar(label="Μέγιστος πληθυσμός")         # χρωματική μπάρα
    plt.xlabel("r (ρυθμός ανάπτυξης)")               # ετικέτα x
    plt.ylabel("P0 (αρχικός πληθυσμός)")             # ετικέτα y
    plt.title("Λογιστικό: heatmap μέγιστου πληθυσμού")  # τίτλος
    plt.tight_layout()                               # περιθώρια
    p = OUT / fname                                  # έξοδος αρχείου
    plt.savefig(p, dpi=170)                          # αποθήκευση
    print("🖼️", p)                                   # ενημέρωση

# ---------------- ΚΥΡΙΟ ΠΡΟΓΡΑΜΜΑ ----------------  # ενότητα
def main():                                          # είσοδος εκτέλεσης
    P = logistic_discrete()                          # εκτέλεση λογιστικού μοντέλου
    pd.DataFrame({"t": np.arange(len(P)), "P": P}).to_csv(OUT / "logistic.csv", index=False)  # αποθήκευση CSV
    plot_series(np.arange(len(P)), P, "Λογιστικό (με θόρυβο σ≈2.5)", "logistic.png")          # γράφημα λογιστικού

    t, prey, pred = lotka_volterra()                 # εκτέλεση Lotka–Volterra
    pd.DataFrame({"t": t, "prey": prey, "pred": pred}).to_csv(OUT / "lv.csv", index=False)   # αποθήκευση CSV
    plot_lv_time(t, prey, pred, "lv_time.png")       # χρονικό γράφημα LV
    plot_lv_phase(prey, pred, "lv_phase.png")        # φασικό διάγραμμα LV

    P0_vals = np.linspace(5, 60, 12)                 # πλέγμα αρχικών πληθυσμών
    r_vals  = np.linspace(0.1, 0.9, 13)              # πλέγμα ρυθμών ανάπτυξης
    Z = sweep_logistic(P0_vals, r_vals, K=900)       # σάρωση και μετρική κορυφής
    pd.DataFrame(Z, index=np.round(P0_vals,2), columns=np.round(r_vals,2)).to_csv(OUT/"logistic_heatmap.csv")  # CSV heatmap
    plot_heatmap(P0_vals, r_vals, Z, "logistic_heatmap.png")   # σχεδίαση heatmap

    print("✅ Ολοκλήρωση: δείτε τα PNG/CSV στον φάκελο", OUT)  # μήνυμα ολοκλήρωσης

if __name__ == "__main__":   # τυπικό guard εκτέλεσης
    main()                   # κλήση κύριας ρουτίνας
