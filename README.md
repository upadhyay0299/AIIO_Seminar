Unsupervised Semantic Anomaly Detection of Retracted CS Papers

This repository contains the code accompanying the MSc seminar paper:

“Unsupervised Semantic Anomaly Detection of Retracted Computer Science Papers”
Master’s in Data and Knowledge Engineering (DKE), OVGU.

The project evaluates whether retracted computer science papers exhibit semantic anomalies relative to non-retracted publications using fully unsupervised learning, based on Sentence-BERT embeddings and classical anomaly detection models (LOF and Isolation Forest).

Requirements

Python 3.8+

Install required packages:

pip install numpy pandas scikit-learn matplotlib seaborn umap-learn sentence-transformers plotly

Usage

Prepare datasets

python fake_paper_detection.py


Run experiments

python main.py


The code trains anomaly detectors only on non-retracted arXiv papers and evaluates performance on retracted papers from the Retraction Watch Database.

Output

Running the code generates:

ROC and Precision–Recall curves

Anomaly score distributions

UMAP semantic visualizations

CSV files for error analysis
