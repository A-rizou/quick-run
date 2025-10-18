import pandas as pd

# Διάβασε το CSV που έχεις ήδη
df = pd.read_csv("random_temperatures.csv")

# Αποθήκευσε το ίδιο αρχείο σε μορφή Excel (.xlsx)
df.to_excel("random_temperatures.xlsx", index=False)

print("✅ Το αρχείο 'random_temperatures.xlsx' δημιουργήθηκε με επιτυχία!")
