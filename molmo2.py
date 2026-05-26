import pandas as pd
import torch
import os
import argparse
from PIL import Image
import PIL.Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoProcessor, AutoModelForImageTextToText

# -------------------------------------------------------------------
# 1. Hilfsfunktionen
# -------------------------------------------------------------------
def create_zero_shot_prompt(hypothesis):
    """
    Erstellt den exakten Prompt für das Modell.
    Wir zwingen das Modell durch die Instruktion dazu, 
    möglichst nur das gewünschte Label auszugeben.
    """
    return (
        f"You are an expert in the Visual Entailment task.\n"
        f"Your task is to evaluate the logical relationship between the provided image and a given hypothesis: '{hypothesis}'.\n\n"
        f"Use the following strict definitions for your classification:\n"
        f"- 'entailment': The hypothesis is TRUE based on the visual evidence in the image.\n"
        f"- 'contradiction': The hypothesis is FALSE based on the visual evidence in the image.\n"
        f"- 'neutral': There are not enough visual cues in the image to make a decision about the truth of the hypothesis.\n\n"
        f"You MUST format your response exactly like this:\n"
        f"Label: [exactly one of: 'entailment', 'neutral', or 'contradiction']"
    )

def create_few_shot_prompt(hypothesis, variation):
    # Punkt am Ende entfernen, um sauber anzuhängen
    shot_h1 = "Woman is playing on a cello while two girls play the banjo"
    shot_h2 = "A baby is with a man"
    shot_h3 = "Africans are being forced to mine for diamonds"
    
    # Varianten mit sauberen Abständen
    vari_hypothesis = {
        "original": ".",
        "negation": " and blue is not red.",
        "start": "Red is red and red is red and red is red and red is red and red is red and ",
        "end": " and red is red and red is red and red is red and red is red and red is red."
    }
    
    # Hypothesen zusammenbauen
    def build_h(base):
        v = vari_hypothesis[variation]
        if variation == "start":
            return v + base.lower() + "."
        else:
            return base + v

    system_instruction = (
        "You are an expert in the Visual Entailment task.\n"
        "Evaluate the logical relationship between the image and the hypothesis.\n"
        "Definitions:\n"
        f"- 'entailment': The hypothesis is TRUE based on the visual evidence in the image.\n"
        f"- 'contradiction': The hypothesis is FALSE based on the visual evidence in the image.\n"
        f"- 'neutral': There are not enough visual cues in the image to make a decision about the truth of the hypothesis.\n\n"
    )
    
    # Bilder laden (sollten idealerweise außerhalb geladen werden, um Speed zu sparen)
    img1 = PIL.Image.open('/home/z/zhangyon/.cache/kagglehub/datasets/hsankesara/flickr-image-dataset/versions/1/flickr30k_images/flickr30k_images/3240637051.jpg')
    img2 = PIL.Image.open('/home/z/zhangyon/.cache/kagglehub/datasets/hsankesara/flickr-image-dataset/versions/1/flickr30k_images/flickr30k_images/1417941060.jpg')
    img3 = PIL.Image.open('/home/z/zhangyon/.cache/kagglehub/datasets/hsankesara/flickr-image-dataset/versions/1/flickr30k_images/flickr30k_images/1872000955.jpg')
    
    prompt_parts = [
        system_instruction,
        
        "\n### Examples",
        
        "Example 1:", img1,
        f"Hypothesis: {build_h(shot_h1)}",
        "Label: contradiction",
        
        "Example 2:", img2,
        f"Hypothesis: {build_h(shot_h2)}",
        "Label: entailment",

        "Example 3:", img3,
        f"Hypothesis: {build_h(shot_h3)}",
        "Label: neutral",

        "\n### Current Task",
        "Image: the remaining image",
        f"Hypothesis: {hypothesis}",
        "\nRespond exactly in this format:",
        "Label: [entailment, neutral, or contradiction]"
    ]

    return prompt_parts, 

def create_cot_zero_shot_prompt(hypothesis):
    return (
            f"You are an expert in the Visual Entailment task.\n"
            f"Let's think step by step and make a three-way decision about the truth of the provided hypothesis:'{hypothesis}' based on the given image:\n\n"
            f"- 'entailment': The hypothesis is TRUE based on the visual evidence in the image.\n"
            f"- 'contradiction': The hypothesis is FALSE based on the visual evidence in the image.\n"
            f"- 'neutral': There are not enough visual cues in the image to make a decision about the truth of the hypothesis.\n\n"
            f"Provide your step-by-step analysis and a classification of the hypothesis.\n"
            f"You MUST format your response exactly like this:\n"
            f"Analysis: [your step-by-step analysis]\n"
            f"Label: [exactly one of: 'entailment', 'neutral', or 'contradiction']"
        )

def parse_model_output(output_text):
    
    analysis = ""
    label = "unknown"
    
    # Prüfen, ob das Modell sich an das Format "Label:" gehalten hat
    if "Label:" in output_text:
        parts = output_text.split("Label:")
        analysis = parts[0].replace("Analysis:", "").strip()
        label_part = parts[1].lower().strip()
    else:
        # Falls das Modell "Label:" vergessen hat, nimm den ganzen Text als Analyse
        analysis = output_text.replace("Analysis:", "").strip()
        label_part = output_text.lower()
    
    # Label extrahieren
    if "entailment" in label_part:
        label = "entailment"
    elif "contradiction" in label_part:
        label = "contradiction"
    elif "neutral" in label_part:
        label = "neutral"

    return analysis, label
# -------------------------------------------------------------------
# 2. Haupt-Evaluations-Funktion
# -------------------------------------------------------------------

def evaluate(dataset_path, image_dir, output_path, strategy, variation):
    model_id = "allenai/Molmo2-4B"
    
    print(f"Lade Datensatz: {dataset_path}")
    df = pd.read_csv(dataset_path,
                    #nrows=4
                    )
    
    print(f"Lade Molmo Modell: {model_id}...")
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16, 
        device_map="cuda",
    )
    model.eval()
    
    generated_analysis=[]
    predictions = []
    
    print(f"Starte Klassifikation für {len(df)} Beispiele...")
    
    for index, row in tqdm(df.iterrows(), total=len(df)):
        image_name = str(row['Flickr30kID'])
        img_path = os.path.join(image_dir, image_name)
        image = Image.open(img_path).convert("RGB")

        hypothesis = row['hypothesis']

        #prompt nach Prompt-Strategie auswählen
        content_list = []
        if strategy == "zero_shot": 
            prompt = create_zero_shot_prompt(hypothesis)
            content_list = [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        if strategy == "few_shot": 
            prompt = create_few_shot_prompt(hypothesis, variation)
            for item in prompt:
                if isinstance(item, PIL.Image.Image):
                    content_list.append({"type": "image", "image": item})
                else:
                    content_list.append({"type": "text", "text": item})
            # Das aktuelle Test-Bild hinzufügen
            content_list.append({"type": "image",  "image": image})
            
        if strategy == "cot_zero_shot":
            prompt = create_cot_zero_shot_prompt(hypothesis)
            content_list = [
                {"type": "image",  "image": image},
                {"type": "text", "text": prompt}
            ]

        messages = [{"role": "user", "content": content_list}]

        # 3. Chat-Template anwenden & Inputs vorbereiten
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=300,
                use_cache=True
            )
            
        generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
        output_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        analysis, final_label = parse_model_output(output_text)
        generated_analysis.append(analysis if analysis else "")
        predictions.append(final_label)


    # Ergebnisse speichern
    if generated_analysis:
        df['analysis'] = generated_analysis
    df['predicted_label'] = predictions
    df.to_csv(output_path, index=False)
    print(f"Ergebnisse gespeichert unter: {output_path}")
    
    # Schnelle Auswertung anzeigen
    y_true = df['gold_label'] # Spaltenname ggf. anpassen
    y_pred = df['predicted_label']
        
    acc = accuracy_score(y_true, y_pred)
    print("\n" + "="*50)
    print(f"ERGEBNISSE FÜR MOLMO-2-4B")
    print("="*50)
    print(f"Accuracy: {acc:.4f}")
    print("\nKlassifikationsbericht:")
    print(classification_report(y_true, y_pred, labels=['entailment', 'neutral', 'contradiction']))

# -------------------------------------------------------------------
# 3. CLI
# -------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Molmo-2-4B on e-SNLI-VE")
    parser.add_argument("--dataset", type=str, required=True, help="Pfad zur CSV (z.B. dev.csv)")
    parser.add_argument("--image_dir", type=str, required=True, help="Ordner mit Flickr30k Bildern")
    parser.add_argument("--strategy", type=str, required=True, choices=["zero_shot", "few_shot", "cot_zero_shot"], help="Prompt-Strategie")
    parser.add_argument("--variation", type=str, required=True, choices=["original", "negation", "start", "end"], help="Datensatz Variation")
    parser.add_argument("--output", type=str, required=True, help="Pfad für die Ergebnis-CSV")
    
    args = parser.parse_args()
    evaluate(args.dataset, args.image_dir, args.output, args.strategy, args.variation)