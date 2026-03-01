# The Finetuning Landscape - A Map of Modern LLM Training

![](https://substackcdn.com/image/fetch/$s_!_Yph!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff1909018-fbb6-4a19-8eb3-7cce5c333837_1920x1080.png)



# 🇬🇧 English Version – Detailed Summary

---

# Lesson 1 – The Finetuning Landscape: A Map of Modern LLM Training

---

## Introduction

Before discussing finetuning (SFT, LoRA, QLoRA), we need to understand:

* Pretraining
* Transformer architecture
* The three Transformer variants
* The modern LLM training pipeline

This lesson builds the **mental model** required to understand everything that follows.

---

# 1️⃣ The Limitations of RNNs

Before Transformers:

* CNNs dominated vision
* RNNs/LSTMs dominated sequence modeling

Problem:

> Long-range dependencies were difficult to capture.

They compressed entire sequences into a single hidden state.

---

# 2️⃣ Attention Mechanism

Attention was introduced to solve the bottleneck in encoder–decoder RNNs.

Instead of compressing everything:

The decoder can dynamically attend to all encoder states.

Core idea:

* Query (Q)
* Keys (K)
* Values (V)

```
Attention(Q, K, V) = softmax(QKᵀ / √d) V
```

Attention is a weighted lookup mechanism.

---

# 3️⃣ Self-Attention

Each token attends to every other token in the same sequence.

This allows:

* Long-range modeling
* Full parallelization
* Rich contextual embeddings

---

# 4️⃣ Positional Encoding

Attention has no notion of order.

Solution:

Inject positional encodings using sine and cosine functions at different frequencies.

---

# 5️⃣ Multi-Head Attention

Multiple attention heads allow:

* Different representation subspaces
* Specialization (syntax, semantics, long-range)

---

# 6️⃣ The Three Transformer Architectures

### Encoder-only (BERT)

* Bidirectional
* Great for understanding
* Not generative

---

### Encoder–Decoder (T5, BART)

* Sequence-to-sequence
* Cross-attention
* Ideal for translation and summarization

---

### Decoder-only (GPT family)

* Causal self-attention
* Next-token prediction
* Scales extremely well

Dominant architecture today.

---

# 7️⃣ Scaling Laws

Performance improves predictably when increasing:

* Model size
* Data
* Compute

Recent breakthroughs came from scaling.

---

# 8️⃣ The LLM Training Pipeline

Formalized by InstructGPT:

Three stages:

1. Pretraining
2. SFT
3. RLHF

Or two-phase view:

* Pretraining
* Post-training

---

# 9️⃣ Pretraining

Objective:

Causal Language Modeling (CLM)

No labels.
No human supervision.

Just:

```
Raw text → predict next token
```

This is where the model learns language and world knowledge.

---

# 🔟 Base Model

After pretraining:

The model is powerful but not aligned.

Fine-tuning teaches behavior.

---

# 1️⃣1️⃣ Continued Pretraining

Used when:

* Adding domain knowledge
* Supporting new languages
* Adapting to distribution shifts

---

# 🎓 Comprehension Questions

---

### 1️⃣ Why was attention introduced?

To remove the information bottleneck of encoder–decoder RNNs and allow dynamic access to input states.

---

### 2️⃣ Why is self-attention superior to RNNs?

Parallelizable, captures long-range dependencies directly, avoids sequential bottlenecks.

---

### 3️⃣ Why do decoder-only models dominate?

Simple objective, scalable training, emergent capabilities at scale.

---

### 4️⃣ What is the difference between pretraining and SFT?

Pretraining = general language modeling.
SFT = supervised instruction learning.

---

### 5️⃣ Why is pretraining self-supervised?

Because the next token serves as the supervision signal.



---


# 🇫🇷 Résumé – Leçon 1 : *The Finetuning Landscape – A Map of Modern LLM Training*

---

## Introduction

Cette première leçon pose les **fondations conceptuelles indispensables** avant d’aborder le fine-tuning (SFT, LoRA, QLoRA, etc.).

Plot twist : on ne parle pas encore de fine-tuning.

On commence par comprendre :

* Ce qu’est réellement le **pré-entraînement (pretraining)**
* Comment fonctionne l’architecture **Transformer**
* Les différences entre les architectures :

  * Encoder-only
  * Encoder–decoder
  * Decoder-only
* Le pipeline complet d’entraînement des LLM modernes

L’objectif : construire une **carte mentale claire** du paysage des LLM.

---

# 1️⃣ Avant les Transformers : le problème des séquences longues

Avant les Transformers :

* Vision → CNN
* NLP → RNN, LSTM

Les RNN/LSTM fonctionnaient bien… mais souffraient d’un problème majeur :

> ❗ Difficulté à gérer les dépendances longues.

Ils devaient compresser toute la séquence dans un vecteur unique (state).

---

# 2️⃣ L’attention : le vrai tournant

L’attention n’a pas été inventée dans le papier Transformer.

Elle a d’abord été introduite pour améliorer les modèles encoder–decoder en traduction automatique.

### Problème initial

Encoder–decoder classique :

* Encoder → compresse toute la phrase en un vecteur
* Decoder → génère la sortie à partir de ce vecteur

Limitation : perte d’information.

### Solution : Attention

L’attention permet au modèle de :

* Comparer une **query (Q)** à plusieurs **keys (K)**
* Pondérer des **values (V)**
* Produire une somme pondérée

Formule conceptuelle :

```
Attention(Q, K, V) = softmax(QKᵀ / √d) V
```

👉 L’attention permet au modèle de « regarder » dynamiquement différentes parties de l’entrée.

---

# 3️⃣ Self-Attention : la révolution

Au lieu d’une séquence qui regarde une autre séquence :

> Chaque token regarde tous les autres tokens.

Exemple :

```
The cat sat on the mat
```

Le token “sat” peut :

* fortement regarder “cat”
* regarder “mat”
* ignorer “the”

Chaque token génère :

* Q (Query)
* K (Key)
* V (Value)

Puis calcule une nouvelle représentation enrichie.

---

# 4️⃣ Positional Encoding

Problème :

L’attention ne contient aucune notion d’ordre.

Solution :

On injecte une information de position via des fonctions sinus/cosinus à différentes fréquences.

* Basse fréquence → information globale
* Haute fréquence → information fine

Cela permet au modèle de comprendre la structure séquentielle sans récursivité.

---

# 5️⃣ Multi-Head Attention

Une seule tête d’attention = une seule perspective.

Multi-head = plusieurs perspectives en parallèle :

* Relations syntaxiques
* Relations sémantiques
* Dépendances longues
* Dépendances locales

Chaque tête apprend un sous-espace différent.

---

# 6️⃣ Les 3 architectures Transformer

## 🔵 Encoder-only (ex : BERT)

* Attention bidirectionnelle
* Très bon pour :

  * Classification
  * Retrieval
  * Similarité sémantique

Pas conçu pour générer du texte.

Objectif typique : Masked Language Modeling.

---

## 🟣 Encoder–Decoder (ex : T5, BART)

Architecture originale du Transformer.

* Encoder → comprend l’entrée
* Decoder → génère la sortie
* Cross-attention entre les deux

Idéal pour :

* Traduction
* Résumé
* Text-to-text

---

## 🔴 Decoder-only (ex : GPT, ChatGPT, Qwen)

Architecture dominante actuelle.

* Self-attention causale
* Objectif : prédire le prochain token

Avantages :

* Scalabilité
* Simplicité
* In-context learning

👉 C’est l’architecture étudiée dans le reste du cours.

---

# 7️⃣ Scaling Laws

Les performances suivent des lois de puissance :

Augmenter :

* Le nombre de paramètres
* La quantité de données
* Le compute

→ améliore les performances de manière prévisible.

Les grands progrès récents ne viennent pas d’une nouvelle architecture…

Mais du scaling massif des Transformers.

---

# 8️⃣ Le Pipeline d’Entraînement des LLM

Standardisé avec InstructGPT (2022).

Deux visions équivalentes :

### Vue en 3 étapes :

1. Pretraining
2. Supervised Fine-Tuning (SFT)
3. RLHF

### Vue en 2 phases :

* Pretraining
* Post-training (SFT + RLHF)

---

# 9️⃣ Pretraining : le cœur du modèle

C’est là que :

* 90% du compute est dépensé
* Le modèle apprend la langue et le monde

Objectif :

Causal Language Modeling (CLM)

> Prédire le token suivant.

Pas d’instructions.
Pas de labels humains.
Pas d’alignement.

Seulement :

```
texte brut → next token prediction
```

C’est du self-supervised learning.

Le label est déjà dans les données.

---

# 🔟 Base Model vs Model Aligné

Après pretraining :

On obtient un **base model**.

Il sait :

* Compléter du texte
* Reproduire des structures
* Générer du code

Mais il ne sait pas :

* Suivre des instructions
* Refuser des demandes
* Être utile et sûr

C’est là que le fine-tuning entre en jeu.

---

# 1️⃣1️⃣ Continued Pretraining

On ne préentraîne presque jamais from scratch.

Mais on peut faire du **continued pretraining** :

* Ajouter un domaine (médical, juridique)
* Ajouter une langue
* Adapter à une nouvelle distribution

C’est ce qui sera implémenté dans le lab.

---

# 🧠 Synthèse mentale

Pretraining = intelligence brute
Fine-tuning = comportement
Alignment = contrôle

---

# 🎓 Questions de compréhension (avec réponses)

---

### 1️⃣ Pourquoi l’attention a-t-elle été introduite initialement ?

**Réponse :**
Pour résoudre le problème du goulot d’étranglement des modèles encoder–decoder RNN qui devaient compresser toute la séquence dans un seul vecteur. L’attention permet un accès dynamique à toutes les représentations de l’entrée.

---

### 2️⃣ Pourquoi la self-attention est-elle plus puissante que les RNN ?

**Réponse :**
Parce qu’elle :

* Modélise directement les dépendances longues
* Est parallélisable
* Ne souffre pas de propagation séquentielle lente

---

### 3️⃣ Pourquoi les decoder-only dominent-ils aujourd’hui ?

**Réponse :**

* Objectif simple (next-token prediction)
* Exploitation massive des données web
* Très bonne scalabilité
* Capacités émergentes avec le scaling

---

### 4️⃣ Quelle est la différence entre pretraining et SFT ?

**Réponse :**
Pretraining = apprentissage général sur texte brut.
SFT = apprentissage supervisé pour suivre des instructions spécifiques.

---

### 5️⃣ Pourquoi le pretraining est-il self-supervised ?

**Réponse :**
Parce que le label (token suivant) est déjà présent dans la séquence. Aucun humain n’a besoin d’annoter les données.

---


