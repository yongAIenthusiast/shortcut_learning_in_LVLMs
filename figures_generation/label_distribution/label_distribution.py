import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Pfade zu Model-CSVs definieren
model_files = {
    'Qwe3-8B': 'output/qwen/11.2.cot_zero_shot.end.tautologie2.csv',
    'Gem3-4B': 'output/gemma/11.2.cot_zero_shot.end.tautologie2.csv',
}

# Reihenfolge der Labels festlegen (wichtig für die Farben)
labels = ['entailment', 'neutral', 'contradiction']
colors = ["#074670", "#0E9EBB", "#9ACBD0"] 

distributions = []

# 2. Daten laden und Verteilung berechnen
# Zuerst die Ground Truth (nehmen sie aus der ersten Datei)
df_first = pd.read_csv(list(model_files.values())[0])
gt_counts = df_first['gold_label'].value_counts(normalize=True).reindex(labels).fillna(0) * 100
distributions.append({'Model': 'GT', **gt_counts.to_dict()})

# Dann für jedes Modell
for name, path in model_files.items():
    df = pd.read_csv(path)
    pred_counts = df['predicted_label'].value_counts(normalize=True).reindex(labels).fillna(0) * 100
    distributions.append({'Model': name, **pred_counts.to_dict()})

# DataFrame für den Plot erstellen
plot_df = pd.DataFrame(distributions).set_index('Model')

# 3.1. Das gestapelte Diagramm erstellen
ax = plot_df.plot(kind='bar', stacked=True, color=colors, figsize=(10, 7), width=0.7)

# Achsen und Titel
plt.title('(d) End', fontsize=14, pad=20)
plt.ylabel('Label Distribution (%)', fontsize=12)
plt.xticks(rotation=0)
plt.ylim(0, 100)

plt.legend(title='Labels', bbox_to_anchor=(1.05, 1), loc='upper left')

# Prozentzahlen in die Balken schreiben
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    if height > 5: # Nur anzeigen, wenn das Segment groß genug ist
        x, y = p.get_xy() 
        ax.text(x + width/2, y + height/2, f'{height:.1f}%', 
                ha='center', va='center', fontsize=10, color='white', fontweight='bold')

plt.tight_layout()
plt.savefig('label_distribution/cot/label_distribution_comparison_end_tau2.1.png', dpi=300)
print("Grafik wurde erfolgreich erstellt.")