import pandas as pd

# Διάβασε το ήδη καθαρισμένο αρχείο
df = pd.read_csv("random_temperatures_clean.csv")

# Εξαγωγή με διαχωριστικό ΤΕΛΕΙΑ-ΚΑΙ-ΚΟΜΜΑ (;)
# που είναι το μόνο που το Google Sheets διαβάζει σωστά στην Ευρώπη
df.to_csv("random_temperatures_clean_semicolon.csv", sep=';', index=False)
print("✅ Δημιουργήθηκε αρχείο: random_temperatures_clean_semicolon.csv")
