import pandas as pd

# Διάβασε το CSV που έχεις ήδη
df = pd.read_csv("dss_report.csv")

# Αποθήκευσε το ίδιο αρχείο σε μορφή Excel (.xlsx)
df.to_excel("dss_report.xlsx", index=False)

print("✅ Το αρχείο 'dss_report.xlsx' δημιουργήθηκε με επιτυχία!")