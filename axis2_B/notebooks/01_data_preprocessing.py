import os
import pandas as pd
from pathlib import Path
from nltk.tokenize.treebank import TreebankWordDetokenizer

# ==========================================
# 路径配置
# ==========================================
ROOT_DIR = Path(r'C:\Users\Lenovo\Desktop\NLP project') 
DATA_DIR = ROOT_DIR / "ebm_nlp_2_00"
OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
detokenizer = TreebankWordDetokenizer()

# ==========================================
# 核心逻辑：提取包含所有统计指标的完整表
# ==========================================

def get_super_full_data():
    all_rows = []
    entities = {'P': 'participants', 'I': 'interventions', 'O': 'outcomes'}
    
    for split in ["train", "test"]:
        id_dir = DATA_DIR / "annotations" / "aggregated" / "hierarchical_labels" / "interventions" / ("test/gold" if split == "test" else "train")
        doc_ids = sorted([p.name.split('.')[0] for p in id_dir.glob("*.ann")])
        
        for d_id in doc_ids:
            tk_path = DATA_DIR / "documents" / f"{d_id}.tokens"
            if not tk_path.exists(): continue
            with open(tk_path, "r", encoding="utf-8") as f:
                tokens = [line.strip() for line in f]
                
            labels = {}
            for key, folder in entities.items():
                subdir = "test/gold" if split == "test" else "train"
                l_path = DATA_DIR / "annotations" / "aggregated" / "hierarchical_labels" / folder / subdir / f"{d_id}.AGGREGATED.ann"
                if not l_path.exists(): l_path = l_path.parent / f"{d_id}.ann"
                if l_path.exists():
                    with open(l_path, "r", encoding="utf-8") as f:
                        labels[key] = [line.strip() for line in f]
            
            if len(labels) < 3: continue # 确保 PICO 都有

            start = 0
            for s_idx, token in enumerate(tokens):
                if token in {".", "!", "?"} or s_idx == len(tokens) - 1:
                    end = s_idx + 1
                    s_tokens = tokens[start : end]
                    s_p = [int(x) for x in labels['P'][start : end]]
                    s_i = [int(x) for x in labels['I'][start : end]]
                    s_o = [int(x) for x in labels['O'][start : end]]
                    
                    # 统计每一类实体的 Token 数量 (即你缺少的那些列)
                    p_count = sum(1 for x in s_p if x != 0)
                    i_count = sum(1 for x in s_i if x != 0)
                    o_count = sum(1 for x in s_o if x != 0)
                    
                    # 判定分类
                    active = sum([p_count > 0, i_count > 0, o_count > 0])
                    if active > 1: g_label = "MIXED"
                    elif p_count > 0: g_label = "P"
                    elif i_count > 0: g_label = "I"
                    elif o_count > 0: g_label = "O"
                    else: g_label = "NONE"
                    
                    all_rows.append({
                        "doc_id": d_id,
                        "sentence_id": len([r for r in all_rows if r['doc_id'] == d_id]),
                        "sentence_start": start,
                        "sentence_end": end,
                        "sentence_text": detokenizer.detokenize(s_tokens),
                        "sentence_tokens": s_tokens,  # <--- 加上这一行，保存原始单词列表
                        "p_count": p_count,
                        "i_count": i_count,
                        "o_count": o_count,
                        "gold_label": g_label
                    })
                    start = end
    return pd.DataFrame(all_rows)

# 执行
super_df = get_super_full_data()
super_df.to_csv(OUTPUT_DIR / "super_full_ebm_data.csv", index=False)
print(f"搞定！总共 {len(super_df)} 行，包含所有详细统计列。")

import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
import sklearn_crfsuite
from nltk.tokenize.treebank import TreebankWordDetokenizer

# ==========================================
# 1. 配置
# ==========================================
ROOT_DIR = Path(r'C:\Users\Lenovo\Desktop\NLP project')
DATA_DIR = ROOT_DIR / "ebm_nlp_2_00"
OUTPUT_DIR = ROOT_DIR / "outputs"
detokenizer = TreebankWordDetokenizer()

# ==========================================
# 2. CRF 特征工程
# ==========================================
def word2features(sent_tokens, i):
    word = str(sent_tokens[i])
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word.isupper()': word.isupper(),
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

# ==========================================
# 3. 数据加载与预处理
# ==========================================
def build_accurate_data():
    print(">>> [Step 1/5] 正在加载原始数据并提取词级标签...")
    all_rows = []
    for split in ["train", "test"]:
        subdir = "test/gold" if split == "test" else "train"
        id_path = DATA_DIR / "annotations" / "aggregated" / "hierarchical_labels" / "interventions" / subdir
        doc_ids = sorted([p.name.split('.')[0] for p in id_path.glob("*.ann")])
        for d_id in doc_ids:
            with open(DATA_DIR / "documents" / f"{d_id}.tokens", "r", encoding="utf-8") as f:
                tokens = [line.strip() for line in f]
            ann_file = id_path / f"{d_id}.AGGREGATED.ann"
            if not ann_file.exists(): ann_file = id_path / f"{d_id}.ann"
            with open(ann_file, "r", encoding="utf-8") as f:
                i_labels = [line.strip() for line in f]
            start = 0
            for s_idx, token in enumerate(tokens):
                if token in {".", "!", "?"} or s_idx == len(tokens) - 1:
                    end = s_idx + 1
                    s_tokens = tokens[start:end]
                    s_labels = i_labels[start:end]
                    bio_labels = ['I' if label != '0' else 'O' for label in s_labels]
                    all_rows.append({
                        "doc_id": d_id,
                        "split": split,
                        "sentence_text": detokenizer.detokenize(s_tokens),
                        "sentence_tokens": s_tokens,
                        "bio_labels": bio_labels,
                        "has_i": 1 if 'I' in bio_labels else 0
                    })
                    start = end
    return pd.DataFrame(all_rows)

full_df = build_accurate_data()
train_df = full_df[full_df['split'] == 'train']
test_df = full_df[full_df['split'] == 'test']

# ==========================================
# 4. 训练模型 (Stage 1 & 2)
# ==========================================
print("\n>>> [Step 2/5] 训练 Stage 1 分类器 (Random Forest)...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
X_train_s1 = embed_model.encode(train_df['sentence_text'].tolist(), show_progress_bar=True)
rf_clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_clf.fit(X_train_s1, train_df['has_i'])

print("\n>>> [Step 3/5] 训练 Stage 2 提取器 (CRF)...")
crf_train_df = train_df[train_df['has_i'] == 1]
X_crf = [[word2features(row['sentence_tokens'], i) for i in range(len(row['sentence_tokens']))] for _, row in crf_train_df.iterrows()]
y_crf = [row['bio_labels'] for _, row in crf_train_df.iterrows()]
crf_model = sklearn_crfsuite.CRF(algorithm='lbfgs', max_iterations=100, all_possible_transitions=True)
crf_model.fit(X_crf, y_crf)

# ==========================================
# 5. 推理与结果汇总 (Step 4 & 5)
# ==========================================
print("\n>>> [Step 4/5] 正在对测试集进行预测...")
X_test_s1 = embed_model.encode(test_df['sentence_text'].tolist(), show_progress_bar=True)
test_df = test_df.copy()
test_df['pred_has_i'] = rf_clf.predict(X_test_s1)

print("\n>>> [Step 5/5] 正在汇总短语并生成最终报表...")
final_results = []

for d_id, group in test_df.groupby('doc_id'):
    found_phrases = set()
    
    for _, row in group[group['pred_has_i'] == 1].iterrows():
        feats = [word2features(row['sentence_tokens'], i) for i in range(len(row['sentence_tokens']))]
        tags = crf_model.predict_single(feats)
        
        # 核心逻辑：聚合连续的 'I' 标签为短语
        current_phrase = []
        for i, tag in enumerate(tags):
            if tag == 'I':
                current_phrase.append(row['sentence_tokens'][i])
            else:
                if current_phrase:
                    # 将列表转为文本，并清理前后空格
                    p_text = detokenizer.detokenize(current_phrase).strip()
                    # 过滤策略：必须包含字母且长度 > 1
                    if any(c.isalpha() for c in p_text) and len(p_text) > 1:
                        found_phrases.add(p_text)
                    current_phrase = []
        
        # 处理句尾残留
        if current_phrase:
            p_text = detokenizer.detokenize(current_phrase).strip()
            if any(c.isalpha() for c in p_text) and len(p_text) > 1:
                found_phrases.add(p_text)
    
    final_results.append({
        "doc_id": d_id,
        "participants_pred": "", # Axis 2-B 默认留空，或根据需要添加 P 模型
        "interventions_pred": "; ".join(sorted(list(found_phrases))),
        "outcomes_pred": ""
    })

# 保存
output_path = OUTPUT_DIR / "AITA_CW_Grp37_Axis2B_Results_Final.csv"
pd.DataFrame(final_results).to_csv(output_path, index=False)
print(f"\n✨ 任务圆满完成！整洁的结果已存至: {output_path}")