import pandas as pd
import numpy as np

# Lade die Datei
df = pd.read_csv('e-ViL/data/esnlive_test.csv', index_col=0)

print(f"Original Einträge: {len(df)}")
print(f"Unique Bilder: {df['Flickr30kID'].nunique()}")

# Gruppiere nach Bild
grouped = df.groupby('Flickr30kID')

# Finde Bilder mit min. 3 verschiedenen Labels (entailment, contradiction, neutral)
valid_images = []
for img_id, group in grouped:
    labels = set(group['gold_label'].values)
    if len(labels) >= 3 and 'entailment' in labels and 'contradiction' in labels and 'neutral' in labels:
        valid_images.append(img_id)

print(f"Bilder mit allen 3 Labels: {len(valid_images)}")

# Zufällig 1000 Bilder auswählen
np.random.seed(42)
selected_images = np.random.choice(valid_images, size=min(1000, len(valid_images)), replace=False)

print(f"Ausgewählte Bilder: {len(selected_images)}")

# Für jedes ausgewählte Bild: nimm je einen Eintrag pro Label
result_rows = []
for img_id in selected_images:
    group = df[df['Flickr30kID'] == img_id]
    
    for label in ['entailment', 'contradiction', 'neutral']:
        label_rows = group[group['gold_label'] == label]
        if len(label_rows) > 0:
            # Wähle zufällig einen Eintrag aus
            random_idx = np.random.choice(len(label_rows))
            result_rows.append(label_rows.iloc[random_idx])

result_df = pd.DataFrame(result_rows)

print(f"Finale Einträge: {len(result_df)}")
print(f"Label Verteilung:\n{result_df['gold_label'].value_counts()}")

# Speichere als neue CSV
result_df.to_csv('e-ViL/data/esnlive_test_filtered.csv')
print("✓ Gespeichert als: e-ViL/data/esnlive_test_filtered.csv")
