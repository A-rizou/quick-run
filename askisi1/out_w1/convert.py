import pandas as pd

# Διάβασε ξεχωριστά κάθε CSV
df_raw  = pd.read_csv("raw.csv")
df_clean = pd.read_csv("clean.csv")
df_diff_mean = pd.read_csv("diff_from_mean_all.csv")
df_diff = pd.read_csv("diff_raw_vs_clean.csv")

# Αποθήκευσε καθένα σε αντίστοιχο XLSX
df_raw.to_excel("raw.xlsx", index=False)
df_clean.to_excel("clean.xlsx", index=False)
df_diff_mean.to_excel("diff_from_mean_all.xlsx", index=False)
df_diff.to_excel("diff_raw_vs_clean.xlsx", index=False)

print("✅ Όλα τα αρχεία μετατράπηκαν επιτυχώς σε .xlsx!")
