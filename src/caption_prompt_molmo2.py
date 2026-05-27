import pandas as pd
import torch
import os
import argparse
from PIL import Image
import PIL.Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoProcessor, AutoModelForImageTextToText
import re
# -------------------------------------------------------------------
# 1. Hilfsfunktionen für kombinierten Output
# -------------------------------------------------------------------

def create_zero_shot_prompt(hypothesis):
    
    return (
        f"You are an expert in the Visual Entailment task.\n"
        f"Your task is to evaluate the logical relationship between the provided image and a given hypothesis.\n\n"
        f"Use the following strict definitions for your classification:\n"
        f"- 'entailment': The hypothesis is TRUE based on the visual evidence in the image.\n"
        f"- 'contradiction': The hypothesis is FALSE based on the visual evidence in the image.\n"
        f"- 'neutral': There are not enough visual cues in the image to make a decision about the truth of the hypothesis.\n\n"
        f"Step 1: Simply state and describe what is shown in the image.\n"
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
    label_match = re.search(r"(?i)Label:\s*(\w+)", output_text)

    if caption_match:
        caption = caption_match.group(1).strip()


    if label_match:
        raw_label = label_match.group(1).lower()
        # Validierung des Labels
        if "entailment" in raw_label:
            label = "entailment"
        elif "contradiction" in raw_label:
            label = "contradiction"
        elif "neutral" in raw_label:
            label = "neutral"

    return caption, analysis, label

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
    captions = []
    
    print(f"Starte Captioning & Klassifikation für {len(df)} Beispiele...")
    
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


        messages = [{"role": "user", "content": content_list}]

        # 2. Processing
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
                max_new_tokens=600, 
                use_cache=True
            )
            
        generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
        output_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        final_caption, analysis, final_label = parse_model_output(output_text)
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
    print(f"\nAccuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, labels=['entailment', 'neutral', 'contradiction']))

# -------------------------------------------------------------------
# 3. CLI
# -------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--strategy", type=str, required=True, choices=["zero_shot", "few_shot", "cot_zero_shot"], help="Prompt-Strategie")
    parser.add_argument("--variation", type=str, required=True, choices=["original", "negation", "start", "end", "no"], help="Datensatz Variation")
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()
    
    evaluate(args.dataset, args.image_dir, args.output, args.strategy, args.variation)