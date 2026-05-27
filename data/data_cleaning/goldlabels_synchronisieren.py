import tlc
import pandas as pd
import os
import glob

# ==========================================
# 1. KONFIGURATION
# ==========================================
TABLE_URL = "/home/z/zhangyon/.local/share/3LC/projects/Visual_Entailment_3LC/datasets/VE_3lc_Model_Comparison_final/tables/SetGoldLabelIn103RowsTo3Values"
LOKALER_ORDNER = "e-ViL/tmp_data1/" 
CLASS_NAMES = ["entailment", "neutral", "contradiction"]

# ==========================================
# 2. 3LC DATEN LADEN & MAPPING ERSTELLEN
# ==========================================
print("Lade 3LC Tabelle herunter...")
df_3lc = tlc.Table.from_url(TABLE_URL).to_pandas()

# Aus dem langen Pfad ("/home/.../bild123.jpg") nur "bild123.jpg holen",
# damit es perfekt zu lokalenr "Flickr30kID"-Spalte passt.
df_3lc['Flickr30kID'] = df_3lc['image'].apply(lambda x: os.path.basename(x))

# Mapping-Dictionary für schnelles Suchen
label_map = {}
for index, row in df_3lc.iterrows():
    bildname = str(row['Flickr30kID'])
    label_zahl = row['gold_label'] # Ist eine Zahl: 0, 1 oder 2
    
    # Falls lokale CSV lieber Texte hat ("entailment") statt Zahlen (0), umwandeln:
    label_text = CLASS_NAMES[label_zahl] 
    
    # Schlüssel ist die Kombination aus Bild und Text
    such_schluessel = (index, bildname)
    label_map[such_schluessel] = label_text
print(list(label_map.items())[:5])
print(f"{len(label_map)} Einträge aus 3LC geladen.\n")

# ==========================================
# 3. ALLE LOKALEN DATEIEN AKTUALISIEREN
# ==========================================
# Sucht alle CSV-Dateien in Ordner
csv_dateien = glob.glob(os.path.join(LOKALER_ORDNER, "*.csv"))


for csv_pfad in csv_dateien:
    print(f"Prüfe {os.path.basename(csv_pfad)}...")
    df_lokal = pd.read_csv(csv_pfad)
    
    aktualisierte_labels = []
    aenderungs_zähler = 0
    # Statistik-Dictionary: {altes_label: {neues_label: count}}
    label_transitions_statistic = {}
    for label in CLASS_NAMES:
        label_transitions_statistic[label] = {target: 0 for target in CLASS_NAMES}

    for idx, row in df_lokal.iterrows():
        bildname = str(row.get('Flickr30kID'))
        such_schluessel = (idx, bildname)
        if idx <= 5:
            print(such_schluessel)
        # Schauen, ob im 3LC-Mapping ein (geänderten) Eintrag dafür da ist
        if such_schluessel in label_map:
            neues_label = label_map[such_schluessel]
            
            # Nur mitzählen, wenn sich das Label wirklich vom alten unterschieden hat
            altes_label = str(row.get('gold_label')).lower()
            
            # Statistik sammeln
            if altes_label in label_transitions_statistic:
                label_transitions_statistic[altes_label][neues_label] += 1
            
            if neues_label != altes_label:
                aenderungs_zähler += 1
                
            aktualisierte_labels.append(neues_label)
        else:
            # Falls nicht im Dashboard gefunden, behalte das alte lokale Label
            aktualisierte_labels.append(row.get('gold_label'))
            
    # Neue Spalte einfügen und CSV überschreiben
    df_lokal['gold_label'] = aktualisierte_labels
    df_lokal.to_csv(csv_pfad, index=False)
    print(f" -> {aenderungs_zähler} Labels in dieser Datei korrigiert und gespeichert.\n")

    print("=" * 60)
    print("LABEL-KORREKTIONSSTATISTIKEN")
    print("=" * 60)

    for original_label in CLASS_NAMES:
        transitions = label_transitions_statistic[original_label]
        total = sum(transitions.values())
        
        if total == 0:
            print(f"\n'{original_label}': Keine Einträge gefunden.")
            continue
        
        print(f"\nVon '{original_label}' ({total} gesamt):")
        for target_label in CLASS_NAMES:
            count = transitions[target_label]
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  -> '{target_label}': {count:4d} ({percentage:6.2f}%)")

print("Fertig! Alle lokalen Dateien sind nun mit dem 3LC Dashboard synchronisiert.\n")
