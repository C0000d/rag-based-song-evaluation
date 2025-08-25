# RAG-based Song Content Evaluation for Child Safety

## Purpose
This project explores how **Retrieval-Augmented Generation (RAG)** can be adapted for evaluating the **appropriateness of songs for children**.  
Instead of running large language models (LLMs) over entire lyrics for every new song—which is costly and resource-intensive—this pipeline uses **retrieval over predefined child-safety questions and responses**. The system efficiently classifies songs into *safe*, *unsafe*, or *borderline* categories by grounding them in specific evaluation intents.

---

## Techniques Used
- **Transformers (BERT)**: To generate embeddings for lyrics, questions, and responses.
- **Mean Pooling Aggregation**: Sequence-level embeddings are obtained by averaging token embeddings (after removing paddings).
- **t-SNE Visualization**: Used to project high-dimensional embeddings (songs, questions, responses) into 3D for intuitive exploration.
- **Vector Similarity (Dot Product / Cosine)**: To retrieve the most relevant safety-related questions given a new song embedding.
- **Evaluation Metrics**: 
  - Precision@k
  - Recall@k
  - Mean Reciprocal Rank (MRR)  
  for analyzing retrieval quality.

---

## Key Features
- **Question-Answer Framing**: Predefined **child-safety related questions** (violence, profanity, weapons, sex, positivity, education, etc.) paired with **prewritten responses**.
- **Lightweight Retrieval**: Avoids full generative inference by matching songs to embeddings of curated safety queries.
- **Extensible Ground Truth**: Allows labeling songs with relevant question indices for evaluation.
- **Visualization Support**: 3D t-SNE plots show separation of songs, questions, and responses in embedding space.
- **Evaluation Suite**: Built-in functions to compute retrieval effectiveness.

---

## Problem Solved
LLMs are powerful but **expensive to run at scale**. Evaluating every song lyric through GPT-style models is infeasible for real-time platforms.  

This project demonstrates:
- How RAG can be adapted to **resource-limited settings** (e.g., running locally on **Apple MPS / CPU**).
- How **semantic retrieval** can act as a cost-effective filter before using heavier models.
- How to **standardize classification** by anchoring judgments in predefined child-safety intents, avoiding ambiguous black-box LLM outputs.

---

## Results
Example retrieval results for test songs:

- **Bullet in the Head** → Retrieved violence, weapons, profanity intents.  
- **Sesame Street** → Retrieved positivity + educational intents.  
- **Barney** → Retrieved educational + uplifting intents.  
- **Straight Outta Compton** → Retrieved violence, profanity, sexual content.  

**Macro Averages**

| P@1   | R@1   | P@5   | R@5   | MRR   |
|-------|-------|-------|-------|-------|
| 1.000 | 0.362 | 0.550 | 0.887 | 1.000 |
Evaluation (Precision@k, Recall@k, MRR) confirms that the model is able to retrieve **correct safety-related intents** for each song with high accuracy, despite using only lightweight BERT embeddings.</br>

---

## Reflection
- **Strengths**:  
  - Efficient on limited hardware (MacBook with CPU/MPS).  
  - Retrieval-first design dramatically reduces cost compared to full generative inference.  
  - Transparent: system outputs both a **label** and the **reasoning evidence** (top retrieved questions + responses).  

- **Limitations**:  
  - Relies on quality of predefined questions/responses; domain adaptation is key.  
  - Mean pooling embeddings may miss nuanced context (future work: CLS pooling or Sentence-BERT).  
  - Requires balanced coverage of **safe vs unsafe examples** to avoid bias.  

- **Future Directions**:  
  - Replace mean-pooled BERT with **sentence-transformers** for stronger retrieval quality.  
  - Add a **final classifier** on top of retrieved evidence for more robust labels.  
  - Explore hybrid pipelines: use RAG retrieval as a *pre-filter*, then escalate borderline songs to a full LLM for deeper inspection.


