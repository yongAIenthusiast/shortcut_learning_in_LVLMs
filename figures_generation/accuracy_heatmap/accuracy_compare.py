import pandas as pd
import numpy as np

# 1. Daten aus image_485694.png übertragen
data = {
    'Model': ['Qwen3-VL-8B-Instruct', 'gemma-3-4b-it'],
    'Standard': [0.79, 0.64],
    'Negation': [0.67, 0.48],
    'Position Start': [0.56, 0.55],
    'Position End': [0.49, 0.51]
}

df = pd.DataFrame(data).set_index('Model')

# 2. Funktion für das Highlighting definieren
def highlight_decrease(row):
    # Der Standard-Wert der Zeile
    standard_val = row['Standard']
    # Berechne Differenz (Rückgang) für jede Zelle
    # Ein größerer Rückgang soll ein dunkleres Blau erzeugen
    diffs = standard_val - row
    
    # Farben erstellen (nur für Spalten außer 'Standard')
    colors = []
    for col_name, val in row.items():
        if col_name == 'Standard':
            colors.append('') # Keine Farbe für Standard
        else:
            decrease = standard_val - val
            # Skalierung der Intensität (0.0 bis 1.0)
            # Wir nehmen an, 0.4 Rückgang ist das Maximum für tiefes Blau
            alpha = min(decrease / 0.4, 1.0) 
            colors.append(f'background-color: rgba(0, 115, 207, {alpha:.2f})')
            
    return colors

# 3. Styling anwenden
styled_df = df.style.apply(highlight_decrease, axis=1)\
    .format("{:.2f}")\
    .set_table_styles([
        {'selector': 'th', 'props': [('border-bottom', '2px solid black'), ('text-align', 'left')]},
        {'selector': 'table', 'props': [('border-collapse', 'collapse'), ('font-family', 'serif')]}
    ])

# 4. Als HTML speichern (kann in Excel oder Word kopiert werden)
styled_df.to_html("vergleichs_tabelle_tau2_cotshot.html")

print("Tabelle wurde als 'vergleichs_tabelle.html' gespeichert.")