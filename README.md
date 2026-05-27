# Investigating Shortcut Learning in Vision–Language Models for Visual Entailment 

This repository contains the program code and project framework for the paper "Investigating Shortcut Learning in Vision–Language Models for Visual Entailment". 
The project evaluates how modern Large Vision-Language Models (LVLMs) process adversarial text modifications (tautologies) when performing Visual Entailment (VE) tasks, examining whether classifications are driven by superficial text-based shortcuts learning or genuine cross-modal grounding. 

## Abstract
Tasks requiring the integration of textual and visual information are now ubiquitous, driving a massive surge of interest in Large Vision-Language Models (LVLMs), which have demonstrated astonishing capabilities across a wide range of multimodal tasks. However, these models frequently rely on dataset biases as shortcuts for prediction, a tendency that can significantly impair their robustness and generalization capabilities. To systematically investigate this vulnerability, this paper introduces the Shortcut Suite, a comprehensive evaluation framework designed to analyze the impact of structural shortcuts on LVLMs' performance by incorporating three distinct shortcut types across four prompting strategies. Extensive experiments yield several key findings, demonstrating first that LVLMs exhibit a varying yet substantial reliance on shortcuts in downstream tasks, which leads to a severe degradation in their performance. Second, the specific choice of the appended tautology actively determines the severity of the model's shortcut reliance. Third, forcing the model to generate a detailed caption of the image premise fails to provide the expected resistance against the shortcut effect. Fourth, despite demonstrating a near-perfect understanding of the source images, the models systematically prefer to rely on simpler textual shortcut patterns, ultimately suppressing valid visual features. Finally, the implementation of few-shot In-Context Learning (ICL) proves significantly more effective at mitigating shortcut reliance than the zero-shot Chain-of-Thought (CoT) strategy. Collectively, these findings offer valuable new insights for evaluating robustness and generalization in vision-language architectures and suggest potential future directions for reducing their reliance on spurious patterns, with all code made publicly available on GitHub.


## Repository Structure

Hardware 
Environment Setup and In-stallation (export file, Dependency ect.)
Data cleaning: url from 3lc
Model Evaluation Pipeline 
Caption Metric
