###Investigating Shortcut Learning in Vision–Language Models for Visual Entailment
├── data/
│   ├── e_snli_ve_subset.csv       # Cleaned balanced sub-dataset (2,973 samples)
│   └── adversarial_variations/    # Generated shortcut variations (Negation, Positions)
├── src/
│   ├── prompt_templates.py        # Implementation of the 4 prompting strategies
│   ├── evaluate_metrics.py        # Custom reference-free jina-clip-v2 pipeline
│   └── inference.py               # Local execution engine for Qwen3, LLaVA, Gemma, Molmo
├── environment.yml                # Anaconda environment file
├── requirements.txt               # Pip dependency list
└── README.md
