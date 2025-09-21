# 🧠 HypER & Friends — Apprentissage de Représentations pour Graphes de Connaissances

> 🎯 *Objectif : Apprendre à représenter les entités (ex: “Paris”, “France”) et les relations (ex: “capitale_de”) sous forme de vecteurs, pour prédire des faits manquants dans un graphe de connaissances.*

Ce projet implémente et compare plusieurs modèles d’**embeddings de graphes de connaissances** (Knowledge Graph Embeddings) :
- **HypER** — Modèle convolutionnel avec filtres hyper-réseaux
- **ConvE** — Modèle convolutionnel 2D
- **DistMult** — Modèle multiplicatif simple
- **ComplEx** — Extension complexe de DistMult (gère les relations asymétriques)
- **HypE** — Version antérieure de HypER

Les modèles sont entraînés et évalués sur des benchmarks standards :
- `FB15k-237` — sous-ensemble filtré de Freebase
- `WN18RR` — sous-ensemble de WordNet sans fuites inverses
- `FB15k`, `WN18` — versions originales (moins recommandées)

---

## 📚 Contexte Scientifique

Dans un **graphe de connaissances**, les faits sont représentés sous forme de triplets :  
> `(sujet, relation, objet)` → ex: `(Paris, capitale_de, France)`

L’objectif est de **prédire l’objet manquant** dans un triplet incomplet :  
> `(Paris, capitale_de, ?)` → doit prédire `France`

C’est ce qu’on appelle la **complétion de liens (link prediction)**.

Les modèles apprennent des **représentations vectorielles** (embeddings) pour chaque entité et relation, puis combinent ces vecteurs pour prédire la probabilité qu’un triplet soit vrai.

---

## 👥 Pour qui est ce projet ?

| Public | Ce qu’il y trouvera |
|--------|----------------------|
| 👩‍🎓 **Étudiants en IA / NLP / Graph ML** | Une implémentation claire de modèles avancés, parfaite pour apprendre ou comparer les architectures. |
| 👨‍🔬 **Chercheurs / Ingénieurs en IA** | Un code fonctionnel, modulaire, facile à étendre ou adapter pour de nouvelles expériences. |
| 👩‍💻 **Développeurs curieux** | Un exemple concret d’apprentissage de représentations sémantiques avec PyTorch. |
| 👔 **Managers / Curieux** | Une démonstration de comment les machines “comprennent” les relations entre concepts du monde réel. |

---

## ⚙️ Fonctionnalités & Modèles Implémentés

### 🧩 Modèles Supportés

| Modèle | Type | Description |
|--------|------|-------------|
| **HypER** | Convolutionnel + Hypernetwork | Génère dynamiquement les filtres de convolution à partir de la relation. Très efficace et léger. |
| **ConvE** | Convolutionnel 2D | Concatène sujet et relation, applique une convolution 2D, puis une couche dense. |
| **DistMult** | Multiplicatif | Score = `sujet ⊙ relation ⋅ objet` — simple mais ne gère pas l’asymétrie. |
| **ComplEx** | Complexe | Extension de DistMult dans l’espace complexe — gère les relations asymétriques (ex: “parent_de”). |
| **HypE** | Convolutionnel paramétré | Chaque relation a ses propres poids de convolution — plus lourd que HypER. |

### 📊 Métriques d’Évaluation

Le modèle est évalué via la tâche de **Link Prediction** :

Pour chaque triplet `(s, r, o)` dans le jeu de test :
- On masque l’objet `o`
- On prédit tous les objets possibles
- On calcule le **rang** de l’objet correct parmi les prédictions

Métriques calculées :
- **Hits@1, Hits@3, Hits@10** → % de fois où la bonne réponse est dans les 1/3/10 premières prédictions
- **Mean Rank (MR)** → rang moyen de la bonne réponse
- **Mean Reciprocal Rank (MRR)** → moyenne de `1/rang` → pénalise fortement les mauvais rangs

---

## 🧩 Technologies & Bibliothèques

```python
import torch
import numpy as np
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
