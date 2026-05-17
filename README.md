# LLM-based-feature-generation

LLM Feature Generation for Movie Budget Classification
Project Overview
This project uses LLM-powered feature engineering to classify movies as high- or low-budget based on their textual descriptions (titles, genres, plots, casts, directors).

Instead of traditional TF-IDF or word embeddings, we use an LLM to discover interpretable features (e.g., "star_power", "production_scale", "franchise_indicator") and then train a standard ML model on those features.

Workflow

Movies Text Data → LLM Discovers Features → LLM Generates Feature Values → Train Classifier → Evaluate
Discovery: LLM analyzes sample texts and proposes distinguishing features

Generation: LLM applies those features to all movies, creating a tabular dataset

Classification: Random Forest/Decision Tree trained on LLM-generated features

Comparison: Baseline TF-IDF model for performance comparison

Key Files Generated
File	Description
discovered_text_features.json	Features discovered by LLM
all_feature_values.csv	Generated feature values for all samples
outputs/	Model results and visualizations
Results
Model	Accuracy
TF-IDF + Logistic Regression	~75%
LLM Features + Random Forest	~83%
LLM-discovered features outperformed the baseline by capturing semantic cues about production scale, cast quality, and genre patterns.

Why This Matters
Interpretability: Features like "star_power_high" are human-readable

No manual feature engineering: LLM discovers what matters

Works with small data: Discovery needs only a few examples per class

Technologies
llm-feature-gen - Core library for LLM feature discovery/generation

scikit-learn - ML models and pipelines

qwen2.5 (via Ollama) - Local LLM provider

