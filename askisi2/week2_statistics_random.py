import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("out_w2")
OUT.mkdir(exist_ok=True)

def make_random_data(n=500, mean=25, std=3, seed=42):
    """Δημιουργεί στοχαστικά δεδομένα θερμοκρασίας."""
    rng = np.random.default_rng(seed)
    temps = rng.normal(loc=mean, scale=std, size=n)
    return temps

def compute_statistics(data):
    """Υπολογίζει βασικά στατιστικά μεγέθη."""
    stats = {
        "Μέσος Όρος": np.mean(data),
        "Διάμεσος": np.median(data),
        "Ελάχιστο": np.min(data),
        "Μέγιστο": np.max(data),
        "Διακύμανση (VAR)": np.var(data, ddof=1),
        "Τυπική Απόκλιση (STDEV)": np.std(data, ddof=1)
    }
    return stats

def plot_histogram(data, bins=20):
    """Σχεδιάζει και αποθηκεύει ιστόγραμμα των τιμών."""
    plt.figure()
    plt.hist(data, bins=bins, color="skyblue", edgecolor="black")
    plt.xlabel("Θερμοκρασία (°C)")
    plt.ylabel("Συχνότητα")
    plt.title("Κατανομή Θερμοκρασιών (Στοχαστικά Δεδομένα)")
    plt.grid(alpha=0.3)
    out_path = OUT / "histogram_temperature.png"
    plt.savefig(out_path, dpi=150)
    print(f"🖼️ Αποθηκεύτηκε γράφημα: {out_path}")

def save_results(data, stats):
    """Αποθηκεύει τα δεδομένα και τα στατιστικά σε CSV και JSON."""
    df_data = pd.DataFrame({"temperature": data})
    df_stats = pd.DataFrame(list(stats.items()), columns=["Δείκτης", "Τιμή"])
    df_data.to_csv(OUT / "random_temperatures.csv", index=False)
    df_stats.to_csv(OUT / "statistics_summary.csv", index=False)
    df_stats.to_json(OUT / "statistics_summary.json", orient="records", force_ascii=False, indent=2)
    print("💾 Αποθηκεύτηκαν τα δεδομένα και τα στατιστικά στον φάκελο out_w2")

def main():
    temps = make_random_data()
    stats = compute_statistics(temps)
    print("\n📊 Βασικά Στατιστικά Θερμοκρασιών\n")
    for k, v in stats.items():
        print(f"{k:25s}: {v:8.3f}")
    plot_histogram(temps)
    save_results(temps, stats)

if __name__ == "__main__":
    main()
