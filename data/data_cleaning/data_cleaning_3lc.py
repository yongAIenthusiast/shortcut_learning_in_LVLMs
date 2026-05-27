#!/usr/bin/env python3
"""
3LC Data Debugging: Modell-Vergleich
Lädt eine Basis-Tabelle in 3LC und fügt die Vorhersagen (predicted_labels) 
von mehreren Modellen aus bestehenden CSV-Dateien hinzu, um falsche Gold-Labels zu finden.
"""

import pandas as pd
import tlc
import os
from pathlib import Path
import argparse
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_NAME = "Visual_Entailment_3LC"
DATASET_NAME = "VE_3lc_Model_Comparison_final"
RUN_NAME = "compare_4_models_final"
RUN_DESCRIPTION = "Vergleich der Vorhersagen von 4 Modellen zur Gold-Label-Korrektur"

CLASS_NAMES = ["entailment", "neutral", "contradiction"]

# ============================================================================
# 1. 3LC TABELLE ERSTELLEN
# ============================================================================

def create_3lc_table_from_csv(csv_path, image_dir):
    print(f"Lade Basis-Datensatz von {csv_path}...")
    df = pd.read_csv(csv_path)
    hypothesis = []
    images = []
    gold_label = []

    forbidden_chars = ['<', '>', '\\', '|', '.', ':', '"', "'", '?', '*', '&']
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Erstelle Basis-Tabelle"):
        image_name = str(row.get('Flickr30kID'))
        img_path = os.path.join(image_dir, image_name)
        images.append(img_path)
        
        raw_hyp = str(row.get('hypothesis'))
        for char in forbidden_chars:
            raw_hyp = raw_hyp.replace(char, "")
        cleaned_hyp = raw_hyp.strip()
        hypothesis.append(cleaned_hyp)

        raw_label = str(row.get('gold_label')).strip().lower()
        if raw_label in CLASS_NAMES:
            gold_label.append(CLASS_NAMES.index(raw_label))
    
    table_data = {
        "image": images,
        "hypothesis_text": hypothesis,
        "gold_label": gold_label
    }

    structure = (
        tlc.PILImage("image"),
        tlc.CategoricalLabel("gold_label", classes=CLASS_NAMES),
    )
    
    tlc_table = tlc.Table.from_dict(
        data=table_data,
        structure=structure,
        dataset_name=DATASET_NAME,
        table_name=DATASET_NAME,
        if_exists="overwrite",
    )
    
    return tlc_table.url 

# ============================================================================
# 2. VORHERSAGEN HOCHLADEN
# ============================================================================

def upload_predictions(run, table_url, result_csvs):
    """
    Nimmt eine Liste von CSV-Dateipfaden, extrahiert die Vorhersagen 
    und pusht sie gebündelt ins Dashboard.
    """
    all_classes = CLASS_NAMES + ["unknown"]
    
    metrics_dict = {}
    schemas_dict = {}

    for csv_path in result_csvs:
        # Modellnamen aus dem Dateinamen extrahieren (z.B. "results_qwen.csv" -> "qwen")
        basename = os.path.basename(csv_path)
        model_name = basename.replace(".csv", "").replace("results_", "")
        col_name = f"pred_{model_name}"
        
        print(f"Lade Vorhersagen für Modell '{model_name}' aus {basename}...")
        df = pd.read_csv(csv_path)
            
        numeric_results = []
        for label in df['predicted_label']:
            label_str = str(label).lower().strip()
            if label_str in all_classes:
                numeric_results.append(all_classes.index(label_str))
            else:
                numeric_results.append(all_classes.index("unknown"))
                
        # Zum Dictionary hinzufügen
        metrics_dict[col_name] = numeric_results
        schemas_dict[col_name] = tlc.CategoricalLabelSchema(classes=all_classes)

    # Alle Metriken in einem einzigen Aufruf an 3LC senden
    if metrics_dict:
        print("\nÜbertrage alle Vorhersagen ins 3LC Dashboard...")
        run.add_metrics(
            metrics=metrics_dict,
            column_schemas=schemas_dict,
            foreign_table_url=table_url,
        )
        print("Fertig! Die Spalten sind jetzt im Dashboard sichtbar.")
    else:
        print("Keine gültigen Vorhersagen zum Hochladen gefunden.")

# ============================================================================
# HAUPTPROGRAMM & CLI(Command Line Interface )
# ============================================================================

def main(args):
    print("="*70)
    print("3LC MODEL COMPARISON & DATA DEBUGGING")
    print("="*70)
    
    print("\nInitializing 3LC Run...")
    run = tlc.init(
        project_name=PROJECT_NAME,
        run_name=RUN_NAME,
        description=RUN_DESCRIPTION,
        if_exists="overwrite",
    )
    
    print("\nCreating 3LC Base Table...")
    table_url = create_3lc_table_from_csv(
        csv_path=args.dataset,
        image_dir=args.image_dir,
    )
    
    print("\nUploading Model Predictions...")
    upload_predictions(run, table_url, args.result_csvs)
    
    print("\n" + "="*70)
    print("✓ UPLOAD COMPLETE!")
    print(f"Results available in 3LC Dashboard: {PROJECT_NAME}/{RUN_NAME}")
    return run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload and compare multiple model predictions in 3LC")
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        help="Path to the original base CSV (to build the table)"
    )
    parser.add_argument(
        "--image_dir", 
        type=str, 
        required=True, 
        help="Path to images directory"
    )
    parser.add_argument(
        "--result_csvs", 
        nargs='+', 
        required=True, 
        help="List of paths to model result CSVs (e.g. results_qwen.csv results_llava.csv)"
    )
    
    args = parser.parse_args()
    run = main(args)