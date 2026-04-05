import os
import pandas as pd
import numpy as np
import joblib
import nltk
import re
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
import sklearn_crfsuite

# Essential resource downloads
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

# --- 1. Path Configuration ---
BASE_PATH = r'C:\Users\Lenovo\Desktop\NLP project\ebm_nlp_2_00'
DOC_PATH = os.path.join(BASE_PATH, 'documents')
ANN_PATH = os.path.join(BASE_PATH, 'annotations', 'individual', 'phase_2', 'interventions', 'train')

# --- 2. Feature Engineering Engine (Specific for Axis 2/NER) ---
def word2features(sent, i):
    word = sent[i][0]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.istitle()': word.istitle(),
        'word[-3:]': word[-3:],
        'postag': nltk.pos_tag([word])[0][1],
    }
    # Contextual features (Previous word)
    if i > 0:
        features['-1:word.lower()'] = sent[i-1][0].lower()
    else:
        features['BOS'] = True  # Beginning of Sentence
        
    # Contextual features (Next word)
    if i < len(sent) - 1:
        features['+1:word.lower()'] = sent[i+1][0].lower()
    else:
        features['EOS'] = True  # End of Sentence
    return features

# --- 3. Data Preprocessing & Physical Alignment ---
def build_clean_datasets():
    print(">>> Starting physical-level data alignment...")
    sentence_data = []
    bio_data = []
    
    # Map Annotation IDs
    ann_files = {f.split('.')[0]: f for f in os.listdir(ANN_PATH) if f.endswith('.ann')}
    
    for fid, ann_name in ann_files.items():
        doc_file = os.path.join(DOC_PATH, f"{fid}.tokens")
        if not os.path.exists(doc_file): continue
        
        with open(doc_file, 'r', encoding='utf-8') as f:
            tokens = [line.strip() for line in f if line.strip()]
        with open(os.path.join(ANN_PATH, ann_name), 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f if line.strip()]
            
        # Sequence Alignment Check
        if len(tokens) != len(labels): continue

        # Generate BIO Tags (for Axis 2 NER)
        last_l = '0'
        for t, l in zip(tokens, labels):
            tag = 'O'
            if str(l) == '1':
                tag = 'B-INT' if last_l == '0' else 'I-INT'
            bio_data.append({'File_ID': fid, 'Word': t, 'Tag': tag})
            last_l = str(l)

        # Generate Sentence-level Labels (for Axis 1 Classifier)
        tmp_sent, has_int = [], False
        for t, l in zip(tokens, labels):
            tmp_sent.append(t)
            if str(l) == '1': has_int = True
            # Split sentences based on punctuation
            if t in ['.', '?', '!']:
                sentence_data.append({
                    'Sentence': " ".join(tmp_sent), 
                    'Label': 1 if has_int else 0
                })
                tmp_sent, has_int = [], False

    # Save processed data for transparency
    pd.DataFrame(sentence_data).to_csv('axis1_data.csv', index=False)
    pd.DataFrame(bio_data).to_csv('axis2_data.csv', index=False)
    
    b_int_count = pd.DataFrame(bio_data)['Tag'].value_counts().get('B-INT', 0)
    print(f"[SUCCESS] Data alignment complete. Total B-INT tags found: {b_int_count}")

# --- 4. Training Pipeline ---
def train_all():
    # Axis 1: Sentence-level Classifier (The Scanner)
    print("\n>>> Training Axis 1 (Sentence Classifier)...")
    df1 = pd.read_csv('axis1_data.csv').dropna()
    pos = df1[df1['Label'] == 1]
    # Balance the dataset using Downsampling
    neg = df1[df1['Label'] == 0].sample(n=min(3000, len(df1)-len(pos)), random_state=42)
    df1_train = pd.concat([pos, neg]).sample(frac=1)
    
    print("[*] Encoding sentences using SBERT...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    X1 = encoder.encode(df1_train['Sentence'].tolist(), show_progress_bar=True)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X1, df1_train['Label'].values)
    joblib.dump(clf, 'axis1_model.pkl')

    # Axis 2: Token-level NER (The Extractor)
    print("\n>>> Training Axis 2 (NER Extractor)...")
    df2 = pd.read_csv('axis2_data.csv').fillna('')
    # Only train on documents that actually contain interventions
    rel_fids = df2[df2['Tag'] != 'O']['File_ID'].unique()
    df2_rel = df2[df2['File_ID'].isin(rel_fids)]
    
    sentences = []
    for _, g in df2_rel.groupby('File_ID'):
        sentences.append(list(zip(g['Word'].values, g['Tag'].values)))
    
    # Apply Oversampling to handle class imbalance
    sentences = sentences * 10 
    X2 = [[word2features(s, i) for i in range(len(s))] for s in sentences]
    y2 = [[l for t, l in s] for s in sentences]
    
    print("[*] Fitting CRF model...")
    crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.01, c2=0.01, max_iterations=100)
    crf.fit(X2, y2)
    joblib.dump(crf, 'axis2_model.pkl')
    print("[SUCCESS] All models trained and saved successfully.")

# --- 5. Integrated Pipeline Inference ---
def run_final_test():
    print("\n>>> Running final integrated inference test...")
    clf = joblib.load('axis1_model.pkl')
    crf = joblib.load('axis2_model.pkl')
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Test Input
    test_doc = "Patients were randomized to receive Aspirin. The control group had no treatment."
    
    sents = nltk.sent_tokenize(test_doc)
    embs = encoder.encode(sents)
    
    # Filter sentences (Lower threshold to 0.1 for higher recall)
    relevance = (clf.predict_proba(embs)[:, 1] >= 0.1).astype(int)
    
    final_extracts = []
    for i, rel in enumerate(relevance):
        if rel == 1:
            tokens = nltk.word_tokenize(sents[i])
            # Format tokens for the feature engine
            feats = [word2features([(t, "") for t in tokens], j) for j in range(len(tokens))]
            tags = crf.predict_single(feats)
            
            # Simple Entity Reconstruction
            for t, tag in zip(tokens, tags):
                if tag != 'O': 
                    final_extracts.append(t)
    
    print("-" * 40)
    print(f"Input Document: {test_doc}")
    print(f"Extracted Interventions: {list(set(final_extracts))}")
    print("-" * 40)

if __name__ == "__main__":
    build_clean_datasets()
    train_all()
    run_final_test()