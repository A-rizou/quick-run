import pandas as pd

# Διάβασε το CSV που έχεις ήδη
df = pd.read_csv("logistic_heatmap.csv")

# Αποθήκευσε το ίδιο αρχείο σε μορφή Excel (.xlsx)
df.to_excel("logistic_heatmap.xlsx", index=False)

print("✅ Το αρχείο 'logistic_heatmap.xlsx' δημιουργήθηκε με επιτυχία!")