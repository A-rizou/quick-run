import pandas as pd

# Διάβασε το CSV που έχεις ήδη
df = pd.read_csv("raw.csv")
df = pd.read_csv("clean.csv")
df = pd.read_csv("diff_from_mean_all.csv")
df = pd.read_csv("diff_raw_vs_clean.csv")

# Αποθήκευσε το ίδιο αρχείο σε μορφή Excel (.xlsx)
df.to_excel("raw.xlsx", index=False)
df.to_excel("clean.xlsx", index=False)
df.to_excel("diff_from_mean_all.xlsx", index=False)
df.to_excel("diff_raw_vs_clean.xlsx", index=False)

print("✅ Το αρχείο 'raw.xlsx' δημιουργήθηκε με επιτυχία!")
print("✅ Το αρχείο 'clean.xlsx' δημιουργήθηκε με επιτυχία!")
print("✅ Το αρχείο 'diff_raw_mean_all.xlsx' δημιουργήθηκε με επιτυχία!")
print("✅ Το αρχείο 'diff_raw_vs_clean.xlsx' δημιουργήθηκε με επιτυχία!")