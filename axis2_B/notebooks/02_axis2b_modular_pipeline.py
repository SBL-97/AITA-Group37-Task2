import pandas as pd
import numpy as np
import ast
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
import sklearn_crfsuite
from nltk.tokenize.treebank import TreebankWordDetokenizer

ROOT_DIR = Path(r'C:\Users\Lenovo\Desktop\NLP project')
INPUT_CSV = ROOT_DIR / "outputs" / "super_full_ebm_data.csv"
OUTPUT_DIR = ROOT_DIR / "outputs"
detokenizer = TreebankWordDetokenizer()

def safe_eval(val):
    """Safely convert string representation of lists back to Python lists."""
    if isinstance(val, list): return val
    try:
        return ast.literal_eval(val)
    except:
        return []

# FEATURE ENGINEERING FOR CRF
def word2features(sent_tokens, i):
    word = str(sent_tokens[i])
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        features['-1:word.lower()'] = str(sent_tokens[i-1]).lower()
    else:
        features['BOS'] = True
    if i < len(sent_tokens) - 1:
        features['+1:word.lower()'] = str(sent_tokens[i+1]).lower()
    else:
        features['EOS'] = True
    return features

# EXECUTION PIPELINE
def run_axis2b_pipeline():
    print(">>> [Step 1/5] Loading preprocessed ground truth data...")
    df = pd.read_csv(INPUT_CSV)
    df['sentence_tokens'] = df['sentence_tokens'].apply(safe_eval)
    
    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    print(">>> [Step 2/5] Initializing Sentence Transformer...")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    final_output = pd.DataFrame({'doc_id': test_df['doc_id'].unique()})
    
    entity_configs = [
        {'col': 'p_count', 'pred': 'participants_pred', 'name': 'Participants'},
        {'col': 'i_count', 'pred': 'interventions_pred', 'name': 'Interventions'},
        {'col': 'o_count', 'pred': 'outcomes_pred', 'name': 'Outcomes'}
    ]

    for config in entity_configs:
        entity_name = config['name']
        target_col = config['col']
        result_col = config['pred']
        
        print(f"\n--- Processing Entity Category: {entity_name} ---")
        
        print(f"Training Stage 1 (RF Classifier) for {entity_name}...")
        y_train_s1 = (train_df[target_col] > 0).astype(int)
        X_train_s1 = embed_model.encode(train_df['sentence_text'].tolist(), show_progress_bar=True)
        
        rf_clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
        rf_clf.fit(X_train_s1, y_train_s1)
        
        print(f"Training Stage 2 (CRF Extractor) for {entity_name}...")
        crf_train_subset = train_df[train_df[target_col] > 0]
        X_crf = [[word2features(row['sentence_tokens'], i) for i in range(len(row['sentence_tokens']))] 
                 for _, row in crf_train_subset.iterrows()]
        y_crf = [['I' if int(target_col == 'p_count' and row['p_count'] > 0) or 
                      (target_col == 'i_count' and row['i_count'] > 0) or 
                      (target_col == 'o_count' and row['o_count'] > 0) else 'O' 
                  for _ in row['sentence_tokens']] for _, row in crf_train_subset.iterrows()]
        
        crf_model = sklearn_crfsuite.CRF(algorithm='lbfgs', max_iterations=100, all_possible_transitions=True)
        crf_model.fit(X_crf, y_crf)
        
        print(f"Predicting {entity_name} in Test Set...")
        X_test_s1 = embed_model.encode(test_df['sentence_text'].tolist(), show_progress_bar=True)
        test_df['s1_prediction'] = rf_clf.predict(X_test_s1)
        
        doc_results = []
        for d_id, group in test_df.groupby('doc_id'):
            found_phrases = set()
            for _, row in group[group['s1_prediction'] == 1].iterrows():
                feats = [word2features(row['sentence_tokens'], i) for i in range(len(row['sentence_tokens']))]
                tags = crf_model.predict_single(feats)
                
                current_phrase = []
                for i, tag in enumerate(tags):
                    if tag == 'I':
                        current_phrase.append(row['sentence_tokens'][i])
                    else:
                        if current_phrase:
                            txt = detokenizer.detokenize(current_phrase).strip()
                            if any(c.isalpha() for c in txt): found_phrases.add(txt)
                            current_phrase = []
                if current_phrase:
                    txt = detokenizer.detokenize(current_phrase).strip()
                    if any(c.isalpha() for c in txt): found_phrases.add(txt)
            
            doc_results.append({'doc_id': d_id, result_col: "; ".join(sorted(list(found_phrases)))})
        
        entity_df = pd.DataFrame(doc_results)
        final_output = final_output.merge(entity_df, on='doc_id', how='left')

    final_output = final_output.fillna("")
    save_path = OUTPUT_DIR / "AITA_CW_Grp37_Axis2B_FINAL_RESULTS.csv"
    final_output.to_csv(save_path, index=False)
    print(f"\n✨ SUCCESS! Pipeline complete. Final report saved to: {save_path}")

if __name__ == "__main__":
    run_axis2b_pipeline()