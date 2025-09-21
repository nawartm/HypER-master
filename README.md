# HypER & Friends — Learning Representations for Knowledge Graphs

> *Objective: Learn to represent entities (e.g., “Paris”, “France”) and relations (e.g., “capital_of”) as vectors, to predict missing facts in a knowledge graph.*

This project implements and compares several **knowledge graph embedding** models:
- **HypER** — Convolutional model with hypernetwork-generated filters
- **ConvE** — 2D convolutional model
- **DistMult** — Simple multiplicative model
- **ComplEx** — Complex-valued extension of DistMult (handles asymmetric relations)
- **HypE** — Earlier version of HypER

Models are trained and evaluated on standard benchmarks:
- `FB15k-237` — Filtered subset of Freebase
- `WN18RR` — Subset of WordNet without inverse relation leakage
- `FB15k`, `WN18` — Original versions (less recommended due to test leakage)

---

## Scientific Background

In a **knowledge graph**, facts are represented as triplets:  
> `(subject, relation, object)` → e.g., `(Paris, capital_of, France)`

The goal is to **predict the missing object** in an incomplete triplet:  
> `(Paris, capital_of, ?)` → should predict `France`

This task is known as **link prediction**.

Models learn **vector representations** (embeddings) for each entity and relation, then combine these vectors to predict the probability that a given triplet is true.

---

## Target Audience

| Audience | What They Will Find |
|----------|----------------------|
| **Students in AI / NLP / Graph ML** | A clear implementation of advanced models — ideal for learning or comparing architectures. |
| **Researchers / AI Engineers** | Functional, modular code, easy to extend or adapt for new experiments. |
| **Curious Developers** | A concrete example of semantic representation learning using PyTorch. |
| **Managers / Non-Technical Readers** | A demonstration of how machines “understand” relationships between real-world concepts. |

---

## Features & Implemented Models

### Supported Models

| Model | Type | Description |
|-------|------|-------------|
| **HypER** | Convolutional + Hypernetwork | Dynamically generates convolutional filters from the relation. Highly effective and lightweight. |
| **ConvE** | 2D Convolutional | Concatenates subject and relation, applies 2D convolution, then a dense layer. |
| **DistMult** | Multiplicative | Score = `subject ⊙ relation ⋅ object` — simple but cannot handle asymmetry. |
| **ComplEx** | Complex-Valued | Extension of DistMult in complex space — handles asymmetric relations (e.g., “parent_of”). |
| **HypE** | Parameterized Convolutional | Each relation has its own convolutional weights — heavier than HypER. |

### Evaluation Metrics

Models are evaluated on the **Link Prediction** task:

For each test triplet `(s, r, o)`:
- Mask the object `o`
- Predict all possible objects
- Compute the **rank** of the correct object among predictions

Metrics computed:
- **Hits@1, Hits@3, Hits@10** → Percentage of times the correct answer is ranked in top 1/3/10
- **Mean Rank (MR)** → Average rank of the correct answer
- **Mean Reciprocal Rank (MRR)** → Mean of `1/rank` → strongly penalizes low-ranked correct answers

---

## Technologies & Libraries

```python
import torch
import numpy as np
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
