import pandas as pd
import torch
from PIL import Image
import PIL.Image
import os
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModelForCausalLM
import re

# -------------------------------------------------------------------
# 1. Konfiguration & Prompt-Design
# -------------------------------------------------------------------

def create_zero_shot_prompt(hypothesis):
    
    return (
        f"You are an expert in the Visual Entailment task.\n"
        f"Your task is to evaluate the logical relationship between the provided image and a given hypothesis.\n\n"
        f"Use the following strict definitions for your classification:\n"
        f"- 'entailment': The hypothesis is TRUE based on the visual evidence in the image.\n"
        f"- 'contradiction': The hypothesis is FALSE based on the visual evidence in the image.\n"
        f"- 'neutral': There are not enough visual cues in the image to make a decision about the truth of the hypothesis.\n\n"
        f"Step 1: State and describe what is shown in the image.\n"
        f"Step 2: Based on the image and the definitions above, evaluate the following hypothesis: '{hypothesis}'.\n\n"
        f"You MUST format your response exactly like this:\n"
        f"Caption: [Your detailed description here]\n"
        f"Label: [exactly one of: 'entailment', 'neutral', or 'contradiction']"
    )
    


def parse_model_output(output_text):
    # Standardwerte
    caption = "not found"
    analysis = ""
    label = "unknown"

    # Regex-Muster: Sucht nach dem Schlüsselwort, ignoriert Groß/Kleinschreibung (?i)
    # und nimmt alles bis zum nächsten Schlüsselwort oder Ende des Strings
    caption_match = re.search(r"(?i)Caption:\s*(.*?)(?=\s*(?:Analysis:|Label:|$))", output_text, re.DOTALL)
    analysis_match = re.search(r"(?i)Analysis:\s*(.*?)(?=\s*Label:|$)", output_text, re.DOTALL)
    label_match = re.search(r"(?i)Label:\s*(\w+)", output_text)

    if caption_match:
        caption = caption_match.group(1).strip()
    
    if analysis_match:
        analysis = analysis_match.group(1).strip()

    if label_match:
        raw_label = label_match.group(1).lower()
        # Validierung des Labels
        if "entailment" in raw_label:
            label = "entailment"
        elif "contradiction" in raw_label:
            label = "contradiction"
        elif "neutral" in raw_label:
            label = "neutral"

    return analysis, caption, label
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
        "gemma": "google/gemma-3-4b-it" 
    }
    
    hf_path = model_paths.get(model_name)
    if not hf_path:
        raise ValueError(f"Modell {model_name} nicht unterstützt. Wähle aus: qwen, llama, gemma")

    # Prozessor und Modell laden (bfloat16 spart 50% VRAM!)
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
            device_map="cuda", # Lädt das Modell automatisch auf die RTX A5000
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
                     #nrows=4
                     )
    
    # Lade Modell
    processor, model = load_model_and_processor(model_name)
    
    generated_analysis=[]
    predictions = []
    captions = []
    
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
       

        messages = [{"role": "user", "content": content_list}]
        
        text_input = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[text_input], images=images_to_process, return_tensors="pt").to("cuda", torch.bfloat16)

        # Inferenz (Generierung)
        with torch.no_grad():
            # Erhöht auf 600, damit das Modell genug Platz für die Caption hat!
            generated_ids = model.generate(**inputs, max_new_tokens=600)
            
        # Output decodieren und bereinigen
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        
        analysis, final_caption, final_label = parse_model_output(output_text)
        generated_analysis.append(analysis if analysis else "")
        captions.append(final_caption) # Neue Liste anlegen!
        predictions.append(final_label)

    # Nach der Schleife:
    if generated_analysis:
        df['analysis'] = generated_analysis
    df['generated_caption'] = captions
    df['predicted_label'] = predictions
    df.to_csv(output_path, index=False)
    
    # 7. Schnelle Auswertung anzeigen
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