# Personalized Skill Gap Analyzer

An interactive Streamlit dashboard that evaluates user skills against job market requirements for specific departments using machine learning (TF-IDF & K-Means Clustering).

## Features
- **Department-Specific Gap Analysis:** Compares user input against aggregated job dataset skills.
- **Machine Learning Clustering:** Uses TF-IDF vectorization and K-Means clustering to align profiles with job clusters.
- **Smart Text Normalization:** Automatically cleans comma-separated skill inputs and removes conjunction artifacts.
- **Exportable Reports:** Generates downloadable CSV summary reports for individual analysis.

## Tech Stack
- **Frontend / Framework:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (TF-IDF Vectorizer, K-Means, Cosine Similarity)

## Local Setup & Run

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/personalized-skill-gap-analyzer.git](https://github.com/YOUR_USERNAME/personalized-skill-gap-analyzer.git)
