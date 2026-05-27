import os
import pandas as pd
import csv
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel
import pandas as pd
import os
import glob

# ---------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------
LOKALER_ORDNER = "e-ViL/tmp_data/" 
IMAGE_DIR = "/home/z/zhangyon/.cache/kagglehub/datasets/hsankesara/flickr-image-dataset/versions/1/flickr30k_images/flickr30k_images"

def main():
    csv_dateien = glob.glob(os.path.join(LOKALER_ORDNER, "*.csv"))

    for csv_pfad in csv_dateien:
        print(f"Rechne für {os.path.basename(csv_pfad)}...")
    
        df = pd.read_csv(csv_pfad)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Nutze Gerät: {device}")
        
        # 1. Native Hugging Face Modelle laden (umgeht den torchmetrics-Bug)
        print("Lade Jina-CLIP Modell...")
        with torch.no_grad():
            model = AutoModel.from_pretrained(
                "jinaai/jina-clip-v2", 
                trust_remote_code=True
            ).to(device)
        
        clip_scores = []
        print(f"Starte Evaluierung für {len(df)} Einträge...")
        
        captions = df['generated_caption'].tolist()
        images = []
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            image_name = str(row['Flickr30kID'])
            img_path = os.path.join(IMAGE_DIR, image_name)
            try:
                image = Image.open(img_path).convert("RGB")
                images.append(image)
            except Exception as e:
                print(f"Fehler beim Laden von {img_path}: {e}")
                images.append(None)

        # Filtern von None-Bildern
        valid_indices = [i for i, img in enumerate(images) if img is not None]
        valid_images = [images[i] for i in valid_indices]
        valid_captions = [captions[i] for i in valid_indices]
        
        if valid_images:
            with torch.no_grad():
                image_embeddings = model.encode_image(valid_images)
                text_embeddings = model.encode_text(valid_captions)
            
            # CLIPScore als Cosinus-Ähnlichkeit berechnen, sum() fuer Skalarprodukt, .item() fuer Float-Ausholung aus dem Tensor
            clip_scores_list = []
            for i in range(len(valid_captions)):
                score = max((text_embeddings[i] * image_embeddings[i]).sum().item(), 0) * 2.5
                clip_scores_list.append(score)
            
            # Scores in Original-Reihenfolge zuordnen
            clip_scores = [None] * len(df)
            for idx, score in zip(valid_indices, clip_scores_list):
                clip_scores[idx] = score
        else:
            clip_scores = [None] * len(df)
        df['clip_score'] = clip_scores
        df.to_csv(csv_pfad, index=False)
        
        # Auswertung
        valid_scores = [s for s in clip_scores if s is not None]
        if valid_scores:
            avg_score = sum(valid_scores) / len(valid_scores)
            print("\n" + "="*50)
            print(f"Erfolgreich bewertet: {len(valid_scores)}/{len(df)} Captions")
            print(f"Durchschnittlicher CLIPScore: {avg_score:.4f}")
            print(f"Ergebnisse gespeichert unter: {csv_pfad}")

if __name__ == "__main__":
    main()