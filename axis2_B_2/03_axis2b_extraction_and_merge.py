import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
import sklearn_crfsuite
from nltk.tokenize.treebank import TreebankWordDetokenizer

# ==========================================
# 1. 核心配置：切换运行目标
# ==========================================
TARGET_ENTITY = 'outcomes'  # 第一次跑设为 'participants'，第二次跑设为 'outcomes'

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

# ==========================================
# 3. 动态数据加载逻辑
# ==========================================
def build_task_data(entity_type):
    print(f"\n>>> [Step 1/5] 正在加载 {entity_type} 的原始数据...")
    all_rows = []
    for split in ["train", "test"]:
        subdir = "test/gold" if split == "test" else "train"
        # 动态定位对应实体的文件夹
        id_path = DATA_DIR / "annotations" / "aggregated" / "hierarchical_labels" / entity_type / subdir
        doc_ids = sorted([p.name.split('.')[0] for p in id_path.glob("*.ann")])
        
        for d_id in doc_ids:
            with open(DATA_DIR / "documents" / f"{d_id}.tokens", "r", encoding="utf-8") as f:
                tokens = [line.strip() for line in f]
            
            ann_file = id_path / f"{d_id}.AGGREGATED.ann"
            if not ann_file.exists(): ann_file = id_path / f"{d_id}.ann"
            with open(ann_file, "r", encoding="utf-8") as f:
                labels_raw = [line.strip() for line in f]

            start = 0
            for s_idx, token in enumerate(tokens):
                if token in {".", "!", "?"} or s_idx == len(tokens) - 1:
                    end = s_idx + 1
                    s_tokens = tokens[start:end]
                    # 关键：将当前实体的非'0'标签统一转为'I'
                    s_bio = ['I' if l != '0' else 'O' for l in labels_raw[start:end]]
                    
                    all_rows.append({
                        "doc_id": d_id,
                        "split": split,
                        "sentence_text": detokenizer.detokenize(s_tokens),
                        "sentence_tokens": s_tokens,
                        "bio_labels": s_bio,
                        "has_target": 1 if 'I' in s_bio else 0
                    })
                    start = end
    return pd.DataFrame(all_rows)

# ==========================================
# 4. 执行流程
# ==========================================
df_all = build_task_data(TARGET_ENTITY)
train_df = df_all[df_all['split'] == 'train']
test_df = df_all[df_all['split'] == 'test']

# Stage 1: 分类
print(f">>> [Step 2/5] 训练 {TARGET_ENTITY} 句子分类器...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
X_train_s1 = embed_model.encode(train_df['sentence_text'].tolist(), show_progress_bar=True)
rf_clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
rf_clf.fit(X_train_s1, train_df['has_target'])

# Stage 2: 提取 (CRF)
print(f">>> [Step 3/5] 训练 {TARGET_ENTITY} CRF 提取器...")
crf_train = train_df[train_df['has_target'] == 1]
X_crf = [[word2features(row['sentence_tokens'], i) for i in range(len(row['sentence_tokens']))] for _, row in crf_train.iterrows()]
y_crf = [row['bio_labels'] for _, row in crf_train.iterrows()]
crf_model = sklearn_crfsuite.CRF(algorithm='lbfgs', max_iterations=100, all_possible_transitions=True)
crf_model.fit(X_crf, y_crf)

# Step 4: 预测
print(f">>> [Step 4/5] 预测测试集 {TARGET_ENTITY}...")
X_test_s1 = embed_model.encode(test_df['sentence_text'].tolist(), show_progress_bar=True)
test_df = test_df.copy()
test_df['pred_active'] = rf_clf.predict(X_test_s1)

# Step 5: 聚合短语
print(f">>> [Step 5/5] 聚合 {TARGET_ENTITY} 短语结果...")
entity_results = []
for d_id, group in test_df.groupby('doc_id'):
    extracted = set()
    for _, row in group[group['pred_active'] == 1].iterrows():
        feats = [word2features(row['sentence_tokens'], i) for i in range(len(row['sentence_tokens']))]
        tags = crf_model.predict_single(feats)
        
        curr_phrase = []
        for i, tag in enumerate(tags):
            if tag == 'I':
                curr_phrase.append(row['sentence_tokens'][i])
            else:
                if curr_phrase:
                    txt = detokenizer.detokenize(curr_phrase).strip()
                    if any(c.isalpha() for c in txt) and len(txt) > 1:
                        extracted.add(txt)
                    curr_phrase = []
        if curr_phrase:
            txt = detokenizer.detokenize(curr_phrase).strip()
            if any(c.isalpha() for c in txt) and len(txt) > 1:
                extracted.add(txt)
    
    entity_results.append({
        "doc_id": d_id,
        f"{TARGET_ENTITY}_pred": "; ".join(sorted(list(extracted)))
    })

# 保存临时文件
temp_file = OUTPUT_DIR / f"temp_{TARGET_ENTITY}_results.csv"
pd.DataFrame(entity_results).to_csv(temp_file, index=False)
print(f"✅ {TARGET_ENTITY} 处理完成！临时文件已存至: {temp_file}")

import pandas as pd
from pathlib import Path

# 路径配置
OUTPUT_DIR = Path(r'C:\Users\Lenovo\Desktop\NLP project\outputs')

# 1. 加载你已经跑出来的 Intervention 结果
df_i = pd.read_csv(OUTPUT_DIR / "AITA_CW_Grp37_Axis2B_Results_Final.csv")
# 删掉可能存在的旧 P/O 列（如果有的话）
df_i = df_i.drop(columns=['participants_pred', 'outcomes_pred'], errors='ignore')

# 2. 加载新生成的 P 和 O 临时文件
df_p = pd.read_csv(OUTPUT_DIR / "temp_participants_results.csv")
df_o = pd.read_csv(OUTPUT_DIR / "temp_outcomes_results.csv")

# 3. 按 doc_id 合并 (使用 outer join 保证不丢 ID)
final_df = df_i.merge(df_p, on='doc_id', how='outer').merge(df_o, on='doc_id', how='outer')

# 4. 整理列顺序与缺失值
final_df = final_df[['doc_id', 'participants_pred', 'interventions_pred', 'outcomes_pred']]
final_df = final_df.fillna("") # 没提出来的填空
final_df = final_df.sort_values('doc_id') # 按ID排序，方便老师对齐

# 5. 保存最终大考卷
final_submission_path = OUTPUT_DIR / "AITA_CW_Grp37_Axis2B_FULL_SUBMISSION.csv"
final_df.to_csv(final_submission_path, index=False)

print(f"🎉 最终成果已合体！请提交此文件：\n{final_submission_path}")
print(final_df.head())