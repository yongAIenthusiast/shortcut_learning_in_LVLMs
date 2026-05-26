import pandas as pd
from upsetplot import from_indicators, plot
import matplotlib.pyplot as plt


model_files = {
    'Gem3-4B-tau1': 'output/gemma/9.zero_shot.end.gemma.csv',
    'Gem3-4B-tau2': 'output/gemma/9.3.zero_shot.end.tautologie2.csv',
}

# Dictionary für die Korrektheits-Indikatoren
correctness_data = {}

# 2. Daten laden und Korrektheit pro Modell bestimmen
for name, path in model_files.items():
    df = pd.read_csv(path)
    
    #die pairID als eindeutigen Schlüssel nutzen
    # Ein True bedeutet: Das Modell hat das Beispiel korrekt klassifiziert
    df['is_correct'] = df['predicted_label'] == df['gold_label']
    
    # Speichere die Korrektheits-Serie mit der pairID als Index
    correctness_data[name] = df.set_index('pairID')['is_correct']

# 3. Alle Daten in einem DataFrame zusammenführen
# führt einen "Inner Join" durch, um sicherzustellen, dass nur 
# Beispiele vergliechen werden, die in allen Dateien vorhanden sind.
comparison_df = pd.DataFrame(correctness_data).dropna()


# 4. UpSet Plot generieren
# from_indicators wandelt die True/False Spalten in das UpSet-Format um
upset_data = from_indicators(comparison_df.columns, comparison_df)

plt.figure(figsize=(12, 8))
plot(upset_data, subset_size='count', show_counts=True)

plt.suptitle('Intersection of Correct Predictions across VLMs', fontsize=16)
plt.savefig('vlm_upset_plot_zeroshot_end_gemma.tau1vstau2.1.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Anzahl der verglichenen Beispiele: {len(comparison_df)}")

'''
# Alle Modelle müssen False sein
none_correct_ids = comparison_df[
    (comparison_df['Qwe3-8B'] == False) & 
    (comparison_df['Mol2-4B'] == False) & 
    (comparison_df['Gem3-4B'] == False) & 
    (comparison_df['LOV1.5-8B'] == False)
].index.tolist()

print(f"IDs der unlösbaren Fälle: {len(none_correct_ids)}")

# Alle Modelle müssen True sein
all_correct_ids = comparison_df[
    (comparison_df['Qwe3-8B'] == True) & 
    (comparison_df['Mol2-4B'] == True) & 
    (comparison_df['Gem3-4B'] == True) & 
    (comparison_df['LOV1.5-8B'] == True)
].index.tolist()

# Nur Molmo2 ist True, alle anderen sind False
molmo_exclusive_ids = comparison_df[
    (comparison_df['Mol2-4B'] == True) & 
    (comparison_df['Qwe3-8B'] == False) & 
    (comparison_df['Gem3-4B'] == False) & 
    (comparison_df['LOV1.5-8B'] == False)
].index.tolist()

molmo_gemma_exclusive_ids = comparison_df[
    (comparison_df['Mol2-4B'] == True) & 
    (comparison_df['Qwe3-8B'] == True) & 
    (comparison_df['Gem3-4B'] == False) & 
    (comparison_df['LOV1.5-8B'] == False)
].index.tolist()

# Dictionary mit den verschiedenen ID-Kategorien erstellen und in CSV speichern
import csv
from itertools import zip_longest

start_ids = {
    'none_correct': none_correct_ids,
    'all_correct': all_correct_ids,
    'molmo_exclusive': molmo_exclusive_ids,
    'molmo_gemma_exclusive': molmo_gemma_exclusive_ids
}

# In CSV speichern (mit zip_longest für unterschiedliche Längen)
with open('start_zero_shot_ids.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(start_ids.keys())
    for row in zip_longest(*start_ids.values(), fillvalue=''):
        writer.writerow(row)



# die end.tautologie2-Daten laden (z.B. von gemma)
df_end.tautologie2 = pd.read_csv('output/gemma/9.zero_shot.end.gemma.csv')

# Jetzt den "unlösbaren" IDs filtern
analysis_df = df_end.tautologie2[df_end.tautologie2['pairID'].isin(none_correct_ids)]

# Speichere die 277 Fälle für deine qualitative Analyse in der Thesis
analysis_df[['pairID', 'hypothesis', 'gold_label', 'explanation']].to_csv('hard_examples_analysis.csv')
'''