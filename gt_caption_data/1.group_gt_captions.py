import pandas as pd

# Lese die gt_captions.txt Datei
df = pd.read_csv('gt_captions.txt')

print(f"Gelesen: {len(df)} Caption-Einträge")
print(f"Spalten: {list(df.columns)}")
print()

# Gruppiere nach Image und fasse alle Captions zusammen
# Verbinde die Captions mit " " und entferne führende/nachfolgende Leerzeichen
grouped = df.groupby('image')['caption'].apply(
    lambda x: ' '.join([str(c).strip() for c in x if pd.notna(c)])
).reset_index()

# Benenne die Spalten um
grouped.columns = ['image_id', 'captions']

print(f"Gruppierte Einträge: {len(grouped)} einzigartige Bilder")
print()

# Speichere die neue CSV-Datei
output_path = 'gt_captions_grouped.csv'
grouped.to_csv(output_path, index=False)

print(f"✓ Datei gespeichert: {output_path}")
print()
print("Erste 3 Zeilen:")
print(grouped.head(3))
