# AITA_CW_Grp37
Introduction to AI and Text Analytics CourseWork Group Repository

# =========================================================
# SB Task2 – EBM-NLP Base Dataset Summary (Team Note)
# =========================================================
#
# Goal:
# - Build a sentence-level dataset from EBM-NLP tokenized abstracts + hierarchical labels
#   to support preliminary clustering (sentence embeddings -> k-means/HAC) and axis testing.
#
# ---------------------------------------------------------
# A) Data location / directory
# ---------------------------------------------------------
# - path: local project root (currently hard-coded in the notebook; teammates should update for their machine)
# - DATA_DIR = Path(f"{path}/ebm_nlp_2_00")
#   Important subfolders:
#   - DATA_DIR / "documents"                          -> *.tokens (tokenized abstracts)
#   - DATA_DIR / "annotations/aggregated/hierarchical_labels/<label_type>/<split>/"
#       where label_type in {"participants","interventions","outcomes"}
#       and split in {"train", "test/gold"}  (note: test is stored under test/gold)
#
# ---------------------------------------------------------
# B) Core helper functions (document-level)
# ---------------------------------------------------------
# - get_doc_ids(split="train", label_type="participants"):
#     Returns doc_id list by reading *_AGGREGATED.ann files under hierarchical_labels/<label_type>/<split>
#     NOTE: if split == "test" it internally uses "test/gold"
#
# - load_labels_for_doc(doc_id, label_type, split):
#     Loads hierarchical labels (0–4) for one doc_id.
#     NOTE: if split == "test" it internally uses "test/gold"
#
# - hierarchical_to_bio(tags) / convert_all_labels_to_bio(labels):
#     Converts hierarchical labels (0–4) into BIO tags (O/B/I) for span-oriented tasks.
#
# - load_document(doc_id):
#     Loads token list from DATA_DIR/documents/{doc_id}.tokens
#
# - load_label_records(doc_ids, label_type, split):
#     Returns a list of dict records per doc:
#       {
#         "doc_id": doc_id,
#         "label_type": label_type,
#         "split": split,
#         "tokens": tokens,
#         "labels_raw": labels_raw,   # hierarchical labels 0–4 (string list)
#         "labels_bio": labels_bio    # BIO tags
#       }
#
# ---------------------------------------------------------
# C) Doc-ID sets used in this notebook
# ---------------------------------------------------------
# 1) Raw doc ids by entity type:
# - doc_ids_p / doc_ids_i / doc_ids_o                 (train; from get_doc_ids)
# - test_doc_ids_p / test_doc_ids_i / test_doc_ids_o  (test; from get_doc_ids)
#
# 2) All-zero filtering (remove docs with no positive labels for a given entity type):
# - count_all_zero_docs(...)
# - all_zero_p / all_zero_i / all_zero_o
# - all_test_zero_p / all_test_zero_i / all_test_zero_o
#
# 3) Modified docs = raw docs minus all-zero docs:
# - modified_doc_ids_p / modified_doc_ids_i / modified_doc_ids_o
# - modified_test_doc_ids_p / modified_test_doc_ids_i / modified_test_doc_ids_o
#
# 4) Common (intersection across P/I/O) – used to ensure same set of docs for P/I/O comparisons:
# - common_modified_doc_ids
# - common_modified_test_doc_ids
#
# ---------------------------------------------------------
# D) Document-level “common records” created (final base inputs)
# ---------------------------------------------------------
# Train:
# - common_participants_records   = load_label_records(common_modified_doc_ids, "participants", "train")
# - common_interventions_records  = load_label_records(common_modified_doc_ids, "interventions", "train")
# - common_outcomes_records       = load_label_records(common_modified_doc_ids, "outcomes", "train")
#
# Test:
# - common_participants_test_records   = load_label_records(common_modified_test_doc_ids, "participants", "test")
# - common_interventions_test_records  = load_label_records(common_modified_test_doc_ids, "interventions", "test")
# - common_outcomes_test_records       = load_label_records(common_modified_test_doc_ids, "outcomes", "test")
#
# For fast lookup:
# - p_record_by_id = {r["doc_id"]: r for r in common_participants_records}
# - i_record_by_id = {r["doc_id"]: r for r in common_interventions_records}
# - o_record_by_id = {r["doc_id"]: r for r in common_outcomes_records}
# (and the analogous *_test_record_by_id for test)
#
# ---------------------------------------------------------
# E) Sentence segmentation (token-span based)
# ---------------------------------------------------------
# - assign_sentence_spans(tokens):
#     Base splitter: split on ".", "?", "!" BUT with decimal protection:
#     do NOT split on digit "." digit  (e.g., 65 . 5)
#     Output: List[(start, end)] where end is exclusive and indexes tokens.
#
# - diagnose_sentence_splitting(records, assign_sentence_spans_function, n_docs=None, short_len=3):
#     Quick sanity check for over-segmentation:
#     prints mean/max sentences per document and mean/max ratio of very short sentences (<=3 tokens).
#
# ---------------------------------------------------------
# F) Sentence text + gold label (sentence-level)
# ---------------------------------------------------------
# - untokenize_tokens(tokens):
#     Uses NLTK TreebankWordDetokenizer to reconstruct a readable sentence_text from sentence_tokens.
#
# - assign_sentence_label(p_count, i_count, o_count):
#     Creates sentence-level gold label from token counts:
#       NONE  : no P/I/O tokens in the sentence
#       MIXED : >=2 of P/I/O present
#       P(articipant) / I(ntervention) / O(utcome) : exactly one field present
#     NOTE: Current notebook returns strings like "P(articipant)" etc.
#           If needed for evaluation, you can map them to {"P","I","O"}.
#
# ---------------------------------------------------------
# G) Final sentence-level dataset builders (main outputs)
# ---------------------------------------------------------
# - load_sentence_records_df(common_doc_ids, p_record_by_id, i_record_by_id, o_record_by_id):
#     Returns pd.DataFrame with columns:
#       - doc_id
#       - sentence_id
#       - sentence_start, sentence_end      (token indices)
#       - sentence_text                     (detokenized)
#       - sentence_tokens                   (List[str])
#       - p_count, i_count, o_count
#       - gold_label                        (NONE / MIXED / P(articipant) / I(ntervention) / O(utcome))
#
# Outputs created in notebook:
# - sentence_records_df       (train sentence-level dataframe)
# - sentence_test_records_df  (test sentence-level dataframe)
#
# Recommended clustering subset (cleaner signal):
# - train_pio = sentence_records_df[sentence_records_df.gold_label.isin(["P(articipant)","I(ntervention)","O(utcome)"])]
# - test_pio  = sentence_test_records_df[sentence_test_records_df.gold_label.isin(["P(articipant)","I(ntervention)","O(utcome)"])]
# (Optionally exclude NONE/MIXED for a clearer schema correspondence sanity check.)
#
# ---------------------------------------------------------
# H) Minimal “how to run” for teammates
# ---------------------------------------------------------
# 1) Update `path` and confirm DATA_DIR points to extracted ebm_nlp_2_00 directory.
# 2) Build doc id lists: doc_ids_p/i/o and test_doc_ids_p/i/o using get_doc_ids.
# 3) Remove all-zero docs: modified_doc_ids_* and modified_test_doc_ids_*.
# 4) Get common intersections: common_modified_doc_ids and common_modified_test_doc_ids.
# 5) Build common_*_records via load_label_records (train + test).
# 6) Build *_record_by_id dicts.
# 7) Build sentence_records_df and sentence_test_records_df via load_sentence_records_df.
# 8) Proceed with embeddings + clustering + axis testing (BERT vs BioBERT, k-means vs HAC, etc.).
# =========================================================