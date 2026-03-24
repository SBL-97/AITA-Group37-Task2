""" AXES: 
I prefer axes 1&2 Axis 1 (Rule-based vs. LLMs) addresses the logic source 
by deciding where you need the rigid, 
predictable accuracy of human-written rules 
versus where you need the flexible, creative intelligence of a language model.
Axis 2 (End-to-End vs. Decomposition) addresses the workflow structure 
by evaluating whether it is better to let the AI handle a request 
in one giant leap or break it down into clear, manageable steps
 (which is more complex to build but much easier to monitor and fix). 
As an additional design axis, we can explore sentence-level and document-level extraction. 
Sentence-level methods process one sentence at a time, which is simple and more precise, 
but may miss information across sentences. 
Document-level methods use the whole abstract, 
so they capture more context, but can be less accurate due to noise. 
Maybe sentence-level has higher precision, and document-level has higher recall.
""" 

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sentence_transformers import SentenceTransformer
from scipy.cluster.hierarchy import dendrogram, linkage

# =========================================================
# STEP 1: DATA LOADING & PREPROCESSING (POS FILTERING)
# =========================================================
doc_path = r'c:/Users/Lenovo/Desktop/NLP project/ebm_nlp_2_00/documents'

file_ids = [f.split('.')[0] for f in os.listdir(doc_path) if f.endswith('.tokens')]
all_filtered_docs = [] 

print(f"[*] Loading {len(file_ids)} clinical documents...")

for fid in file_ids:
    token_p = os.path.join(doc_path, f"{fid}.tokens")
    pos_p = os.path.join(doc_path, f"{fid}.pos")

    with open(token_p, 'r', encoding='utf-8') as f:
        tokens = [line.strip() for line in f if line.strip()]
    with open(pos_p, 'r', encoding='utf-8') as f:
        pos_tags = [line.strip() for line in f if line.strip()]

    # Filter for Nouns and Adjectives to capture PICO entities
    filtered = [t for t, p in zip(tokens, pos_tags) if p.startswith(('NN', 'JJ'))]
    all_filtered_docs.append(" ".join(filtered))

# =========================================================
# STEP 2: FEATURE EXTRACTION (VECTORIZATION)
# =========================================================
print("[*] Generating feature matrices...")

# TF-IDF for keyword extraction later
tfidf_vec = TfidfVectorizer(stop_words='english')
X_m2 = tfidf_vec.fit_transform(all_filtered_docs) 

# SBERT for high-quality semantic clustering
model = SentenceTransformer('all-MiniLM-L6-v2') 
X_m4 = model.encode(all_filtered_docs, show_progress_bar=True)

# =========================================================
# STEP 3: VALIDATION (ELBOW & SILHOUETTE)
# =========================================================
k_range = range(2, 7) # Testing 2 to 6 clusters
elbow_data = []
silhouette_data = []

print("\n>>> Running Task 1: Validation (Elbow & Silhouette Analysis)...")
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_m4)
    
    elbow_data.append(km.inertia_)
    # Silhouette_score is slow on 5k docs, but necessary for report
    score = silhouette_score(X_m4, labels)
    silhouette_data.append(score)
    print(f"Testing K={k}: SSE={km.inertia_:.2f}, Silhouette={score:.4f}")

# Plotting Validation Results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(k_range, elbow_data, 'bo-')
ax1.set_title('Elbow Method (SSE)')
ax2.plot(k_range, silhouette_data, 'ro-')
ax2.set_title('Silhouette Analysis')
plt.savefig('k_validation_results.png')
print("[Output] Validation plots saved as 'k_validation_results.png'")

# =========================================================
# STEP 4: FINAL K-MEANS EXECUTION (K=4)
# =========================================================
BEST_K = 4
RANDOM_SEED = 42

print(f"\n[*] Executing Final K-Means (K={BEST_K}) based on PICO Framework...")
final_kmeans = KMeans(n_clusters=BEST_K, random_state=RANDOM_SEED, n_init=10)
clusters = final_kmeans.fit_predict(X_m4)

# =========================================================
# STEP 5: PICO KEYWORD EXTRACTION
# =========================================================
def extract_top_keywords(tfidf_matrix, labels, vectorizer, n_terms=15):
    feature_names = vectorizer.get_feature_names_out()
    for i in range(BEST_K):
        row_indices = np.where(labels == i)[0]
        cluster_tfidf = tfidf_matrix[row_indices]
        mean_tfidf = cluster_tfidf.mean(axis=0).A1
        top_indices = mean_tfidf.argsort()[::-1][:n_terms]
        top_words = [feature_names[idx] for idx in top_indices]
        print(f"\nCluster {i} Top Keywords:\n" + " | ".join(top_words))

extract_top_keywords(X_m2, clusters, tfidf_vec)

# =========================================================
# STEP 6: HAC COMPARISON & DENDROGRAM
# =========================================================
print("\n>>> Running Hierarchical Clustering (HAC) for Comparison...")
hac_model = AgglomerativeClustering(n_clusters=BEST_K, linkage='ward')
hac_clusters = hac_model.fit_predict(X_m4)

ari_score = adjusted_rand_score(clusters, hac_clusters)
print(f"[Analysis] Adjusted Rand Index (ARI) between K-Means and HAC: {ari_score:.4f}")

# Generate Dendrogram for a representative subset
X_subset = X_m4[:50] 
linked = linkage(X_subset, 'ward')
plt.figure(figsize=(10, 5))
dendrogram(linked)
plt.title('HAC Dendrogram (Subset of 50 Docs)')
plt.savefig('hac_dendrogram.png')

# =========================================================
# STEP 7: RESULTS
# =========================================================
output_df = pd.DataFrame({'File_ID': file_ids, 'KMeans_Cluster': clusters, 'HAC_Cluster': hac_clusters})
output_df.to_csv('final_clustering_report.csv', index=False)
print("\n[Final Output] assignments saved to 'final_clustering_report.csv'")