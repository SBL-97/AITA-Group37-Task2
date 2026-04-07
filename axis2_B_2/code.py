import os
import joblib
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import sklearn_crfsuite

# --- 1. Core Path Configuration ---
# Root directory of your dataset
BASE_DIR = Path(r'C:\Users\Lenovo\Desktop\NLP project\ebm_nlp_2_00')
# Dedicated directory for this specific task
PROJECT_DIR = BASE_DIR / "axis2_B_2"
# Directory for saving outputs (plots and CSVs)
OUTPUT_DIR = PROJECT_DIR / "outputs"

# Create directories if they do not exist
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set the working directory to the project folder
os.chdir(PROJECT_DIR)

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
sns.set_style("whitegrid")

# --- 2. Data Parsing Engine (B-Method Stage 0) ---

def load_ebm_data(data_dir, label_type="interventions"):
    print(f">>> [Data Loading] Reading Aggregated annotations from {data_dir}...")
    label_path = data_dir / f"annotations/aggregated/hierarchical_labels/{label_type}/train"
    doc_ids = [f.name.split('.')[0].split('_')[0] for f in label_path.glob("*.ann")]
    
    sentence_records = []
    bio_records = []
    detokenizer = TreebankWordDetokenizer()

    # Processing 1200 documents to balance training speed and model quality
    for doc_id in doc_ids[:1200]:
        try:
            # Load Tokens
            token_file = data_dir / f"documents/{doc_id}.tokens"
            with open(token_file, 'r', encoding='utf-8') as f:
                tokens = [line.strip() for line in f if line.strip()]
            
            # Load Labels
            ann_file = label_path / f"{doc_id}.AGGREGATED.ann"
            if not ann_file.exists(): 
                ann_file = label_path / f"{doc_id}_AGGREGATED.ann"
            
            with open(ann_file, 'r', encoding='utf-8') as f:
                raw_labels = [line.strip() for line in f if line.strip()]
            
            # Convert hierarchical labels to BIO format
            bio_labels, last = [], '0'
            for l in raw_labels:
                bio_labels.append('O' if l == '0' else ('B-INT' if last == '0' else 'I-INT'))
                last = l
            
            # Store Token-level records (for Stage 2: CRF)
            for t, l in zip(tokens, bio_labels):
                bio_records.append({'File_ID': doc_id, 'Word': t, 'Tag': l})
            
            # Store Sentence-level records (for Stage 1: Classifier)
            start = 0
            for i, t in enumerate(tokens):
                if t in ['.', '!', '?'] or i == len(tokens)-1:
                    sent_tokens = tokens[start:i+1]
                    is_rel = any(tag != 'O' for tag in bio_labels[start:i+1])
                    sentence_records.append({
                        'Sentence': detokenizer.detokenize(sent_tokens),
                        'Label': 1 if is_rel else 0
                    })
                    start = i + 1
        except Exception as e:
            continue
        
    return pd.DataFrame(sentence_records), pd.DataFrame(bio_records)

# --- 3. B-Method Physical Layer Implementation (Stage 1 & 2) ---

def word2features(sent, i):
    word = sent[i][0]
    features = {
        'bias': 1.0, 
        'word.lower()': word.lower(), 
        'word.istitle()': word.istitle(), 
        'postag': nltk.pos_tag([word])[0][1]
    }
    if i > 0: 
        features['-1:word.lower()'] = sent[i-1][0].lower()
    else: 
        features['BOS'] = True
    if i < len(sent) - 1: 
        features['+1:word.lower()'] = sent[i+1][0].lower()
    else: 
        features['EOS'] = True
    return features

def run_b_method_pipeline(df_s, df_b):
    # --- Stage 1: Sentence Classifier (Filter) ---
    print("\n>>> [Stage 1] Building Sentence Classifier (Narrowing Search Space)...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    X1 = encoder.encode(df_s['Sentence'].tolist(), show_progress_bar=True)
    y1 = df_s['Label'].values
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X1, y1)
    y_pred1 = clf.predict(X1)
    
    # --- Stage 2: Span Extractor (Pinpoint) ---
    print("\n>>> [Stage 2] Building Span Extractor (Refined Extraction)...")
    # Training only on documents containing entities to demonstrate modular efficiency
    rel_ids = df_b[df_b['Tag'] != 'O']['File_ID'].unique()
    df_rel = df_b[df_b['File_ID'].isin(rel_ids)]
    sentences = [list(zip(g['Word'].values, g['Tag'].values)) for _, g in df_rel.groupby('File_ID')]
    
    X2 = [[word2features(s, i) for i in range(len(s))] for s in sentences]
    y2 = [[l for t, l in s] for s in sentences]
    
    crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100)
    crf.fit(X2, y2)
    
    # Save models to the dedicated project folder
    joblib.dump(clf, 'axis1_model.pkl')
    joblib.dump(crf, 'axis2_model.pkl')
    
    return clf, crf, y1, y_pred1

# --- 4. Visualization & Metrics Export ---

def export_results(df_s, y_true, y_pred):
    print("\n>>> [Visualization] Generating experimental result reports...")
    
    # 1. Label Distribution Plot
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Label', data=df_s, palette='coolwarm')
    plt.title('Stage 1: Intervention Sentence Distribution')
    plt.savefig(OUTPUT_DIR / 'label_distribution.png', dpi=300)
    
    # 2. Classifier Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title('Stage 1: Classifier Confusion Matrix')
    plt.savefig(OUTPUT_DIR / 'axis2_results.png', dpi=300)
    
    # 3. Quantitative Metrics CSV
    report = classification_report(y_true, y_pred, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(OUTPUT_DIR / 'axis2a_summary.csv')
    
    print(f"\n[Success] All code, models, and results are saved in:\n{PROJECT_DIR}")

if __name__ == "__main__":
    df_sent, df_bio = load_ebm_data(BASE_DIR)
    if not df_sent.empty:
        clf_model, crf_model, y_true, y_pred = run_b_method_pipeline(df_sent, df_bio)
        export_results(df_sent, y_true, y_pred)