import os
import pandas as pd
from pathlib import Path
from nltk.tokenize.treebank import TreebankWordDetokenizer

ROOT_DIR = Path(r'C:\Users\Lenovo\Desktop\NLP project') 
DATA_DIR = ROOT_DIR / "ebm_nlp_2_00"
OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
detokenizer = TreebankWordDetokenizer()

# DATA EXTRACTION & STATISTICS

def get_super_full_data():
    """
    Parses the raw EBM-NLP files (.tokens and .ann) across all three entity types (PICO).
    Generates a sentence-level dataframe with token counts for each entity.
    """
    all_rows = []
    entities = {'P': 'participants', 'I': 'interventions', 'O': 'outcomes'}
    
    for split in ["train", "test"]:
        subdir = "test/gold" if split == "test" else "train"
        
        id_dir = DATA_DIR / "annotations" / "aggregated" / "hierarchical_labels" / "interventions" / subdir
        doc_ids = sorted([p.name.split('.')[0] for p in id_dir.glob("*.ann")])
        
        print(f"Processing {len(doc_ids)} documents in {split} split...")
        
        for d_id in doc_ids:
            tk_path = DATA_DIR / "documents" / f"{d_id}.tokens"
            if not tk_path.exists(): 
                continue
            with open(tk_path, "r", encoding="utf-8") as f:
                tokens = [line.strip() for line in f]
                
            labels = {}
            for key, folder in entities.items():
                l_path = DATA_DIR / "annotations" / "aggregated" / "hierarchical_labels" / folder / subdir / f"{d_id}.AGGREGATED.ann"
                if not l_path.exists(): 
                    l_path = l_path.parent / f"{d_id}.ann"
                
                if l_path.exists():
                    with open(l_path, "r", encoding="utf-8") as f:
                        labels[key] = [line.strip() for line in f]
            
            if len(labels) < 3: 
                continue 

            start = 0
            for s_idx, token in enumerate(tokens):
                if token in {".", "!", "?"} or s_idx == len(tokens) - 1:
                    end = s_idx + 1
                    s_tokens = tokens[start : end]
                    
                    s_p = [int(x) for x in labels['P'][start : end]]
                    s_i = [int(x) for x in labels['I'][start : end]]
                    s_o = [int(x) for x in labels['O'][start : end]]
                    
                    p_count = sum(1 for x in s_p if x != 0)
                    i_count = sum(1 for x in s_i if x != 0)
                    o_count = sum(1 for x in s_o if x != 0)
                    
                    active_categories = sum([p_count > 0, i_count > 0, o_count > 0])
                    if active_categories > 1: 
                        g_label = "MIXED"
                    elif p_count > 0: 
                        g_label = "P"
                    elif i_count > 0: 
                        g_label = "I"
                    elif o_count > 0: 
                        g_label = "O"
                    else: 
                        g_label = "NONE"
                    
                    all_rows.append({
                        "doc_id": d_id,
                        "split": split,
                        "sentence_id": len([r for r in all_rows if r['doc_id'] == d_id]),
                        "sentence_start": start,
                        "sentence_end": end,
                        "sentence_text": detokenizer.detokenize(s_tokens),
                        "sentence_tokens": s_tokens,
                        "p_count": p_count,
                        "i_count": i_count,
                        "o_count": o_count,
                        "gold_label": g_label
                    })
                    start = end
                    
    return pd.DataFrame(all_rows)

if __name__ == "__main__":
    print("Starting Data Preprocessing...")
    super_df = get_super_full_data()
    
    output_file = OUTPUT_DIR / "super_full_ebm_data.csv"
    super_df.to_csv(output_file, index=False)
    
    print(f"Success! Generated {len(super_df)} sentence records.")
    print(f"File saved to: {output_file}")