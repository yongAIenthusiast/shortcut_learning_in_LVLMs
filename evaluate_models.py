import pandas as pd
import torch
from PIL import Image
import PIL.Image
import os
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModelForCausalLM

# -------------------------------------------------------------------
# 1. Konfiguration & Prompt-Design
# -------------------------------------------------------------------

def create_zero_shot_prompt(hypothesis):
  
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
    
    # Bilder laden 
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
        "Image: the remaining image"
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
# 2. Modell-Ladefunktion
# -------------------------------------------------------------------

def load_model_and_processor(model_name):
    """
    Lädt das Modell speichereffizient in bfloat16 auf die GPU.
    """
    print(f"Lade Modell: {model_name}...")
    
    # Mapping von einfachen Namen zu den offiziellen Hugging Face Repositories
    model_paths = {
        "qwen": "Qwen/Qwen3-VL-8B-Instruct",
        "llava_ov": "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
        "gemma": "google/gemma-3-4b-it",
    }
    
    hf_path = model_paths.get(model_name)

    # Prozessor und Modell laden (bfloat16 spart 50% VRAM!)
    # Prozessor laden
    processor = AutoProcessor.from_pretrained(hf_path, use_fast=True, trust_remote_code=True)
    
    
    if "LLaVA" in hf_path:
        model = AutoModelForCausalLM.from_pretrained(
        hf_path,
        torch_dtype=torch.bfloat16,  
        device_map="cuda",
        trust_remote_code=True
    )
    
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            hf_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda", 
            trust_remote_code=True
        )

    
    # Versetze das Modell in den Evaluationsmodus (kein Training)
    model.eval()
    return processor, model

# -------------------------------------------------------------------
# 3. Haupt-Evaluations-Schleife
# -------------------------------------------------------------------

def evaluate(dataset_path, image_dir, model_name, output_path, strategy, variation):
    # Lade Datensatz
    print(f"Lade Datensatz: {dataset_path}")
    df = pd.read_csv(dataset_path, 
                     #nrows=4,
                     )
    
    # Lade Modell
    processor, model = load_model_and_processor(model_name)
    generated_analysis=[]
    predictions = []
    
    print(f"Starte Klassifikation für {len(df)} Beispiele...")
    # tqdm zeigt einen schönen Fortschrittsbalken im Terminal an
    for index, row in tqdm(df.iterrows(), total=len(df)):
        image_name = str(row['Flickr30kID'])
        img_path = os.path.join(image_dir, image_name)
        image = Image.open(img_path).convert("RGB")
    
        hypothesis = row['hypothesis']

        #prompt nach Prompt-Strategie auswählen
        content_list = []
        images_to_process = []
        if strategy == "zero_shot": 
            prompt = create_zero_shot_prompt(hypothesis)
            content_list = [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
            images_to_process = [image]
        if strategy == "few_shot": 
            prompt = create_few_shot_prompt(hypothesis, variation)
            for item in prompt:
                if isinstance(item, PIL.Image.Image):
                    content_list.append({"type": "image"})
                    images_to_process.append(item)
                else:
                    content_list.append({"type": "text", "text": item})
            # Das aktuelle Test-Bild hinzufügen
            content_list.append({"type": "image"})
            images_to_process.append(image) # Alle Bilder sammeln
        if strategy == "cot_zero_shot":
            prompt = create_cot_zero_shot_prompt(hypothesis)
            content_list = [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
            images_to_process = [image]

        messages = [{"role": "user", "content": content_list}]
        
        text_input = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[text_input], images=images_to_process, return_tensors="pt").to("cuda", torch.bfloat16)

        # Inferenz (Generierung)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=300)
            
        # Output decodieren und bereinigen
        # schneiden den Input-Teil ab, um nur die neue Antwort zu bekommen
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        
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
    print(f"ERGEBNISSE FÜR {model_name.upper()} AUF {os.path.basename(dataset_path)}")
    print("="*50)
    print(f"Accuracy: {acc:.4f}")
    print("\nKlassifikationsbericht:")
    print(classification_report(y_true, y_pred, labels=['entailment', 'neutral', 'contradiction']))

# -------------------------------------------------------------------
# 4. CLI Argument-Parser
# -------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LVLMs on e-SNLI-VE datasets")
    parser.add_argument("--dataset", type=str, required=True, help="Pfad zur CSV-Datei (z.B. esnlive_negation.csv)")
    parser.add_argument("--image_dir", type=str, required=True, help="Pfad zum Flickr30k Bildordner")
    parser.add_argument("--model", type=str, required=True, choices=["qwen", "llava_ov", "gemma"], help="Modell auswählen")
    parser.add_argument("--strategy", type=str, required=True, choices=["zero_shot", "few_shot", "cot_zero_shot"], help="Prompt-Strategie")
    parser.add_argument("--variation", type=str, required=True, choices=["original", "negation", "start", "end"], help="Datensatz Variation")
    parser.add_argument("--output", type=str, required=True, help="Pfad für die Ergebnis-CSV")
    
    args = parser.parse_args()
    
    evaluate(args.dataset, args.image_dir, args.model, args.output, args.strategy, args.variation)