# Low-Dimensional Embedding Alignment in LVLMs

This project studies how to align **text** and **vision** embeddings from Large Vision-Language Models (LVLMs) using low-dimensional projections.  
We evaluate cosine similarity, PCA-based alignment, and cross-modal classification with LDA.

## Scripts
- `cosine_sim.py` – compute similarity matrices
- `cross_modal_pca.py` – apply PCA + z-score for alignment
- `cross_modal_lda.py` – train LDA on one modality, test on the other
