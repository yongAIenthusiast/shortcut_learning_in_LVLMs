import pandas as pd

# Lese die esnlive.csv, um die einzigartigen Image-IDs zu finden
print("Lade esnlive.csv...")
esnlive = pd.read_csv('e-ViL/data/esnlive.csv')


image_id_col = 'Flickr30kID'
unique_image_ids = esnlive[image_id_col].unique()

print(f"✓ Gefunden: {len(unique_image_ids)} einzigartige Image-IDs")

# Lese die gt_captions_grouped.csv
print("Lade gt_captions_grouped.csv...")
gt_captions = pd.read_csv('gt_captions_grouped.csv')
print(f"Ursprüngliche Zeilen: {len(gt_captions)}")
print()

# Filtere nach Image-IDs, die in esnlive subdataset vorkommen
# Berücksichtige, dass die Image-IDs in gt_captions mit .jpg enden könnten
print("Filtere Captions nach esnlive subset Image-IDs...")
gt_captions_filtered = gt_captions[gt_captions['image_id'].isin(unique_image_ids)]

print(f"✓ Gefilterte Zeilen: {len(gt_captions_filtered)}")
print()

# Speichere die neue CSV-Datei
output_path = 'gt_captions_esnlive.csv'
gt_captions_filtered.to_csv(output_path, index=False)

print(f"✓ Datei gespeichert: {output_path}")
print()
print("Erste 3 Zeilen:")
print(gt_captions_filtered.head(3))
