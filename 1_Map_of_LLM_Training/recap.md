# The Finetuning Landscape - A Map of Modern LLM Training

![](https://substackcdn.com/image/fetch/$s_!_Yph!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff1909018-fbb6-4a19-8eb3-7cce5c333837_1920x1080.png)


Here is the formatted summary of Lesson 1 in Markdown, separated into English and French versions as requested.

# 🇬🇧 English Summary

## Context and Goals

This first lesson serves as the foundation for the entire bootcamp. Instead of diving straight into finetuning, it covers the essential background needed to understand modern AI.

* **Core Concepts**: Understanding pretraining, the Transformer architecture, and the modern training pipeline from initial training to alignment.
* **Mental Map**: Building a clear conceptual framework so that advanced techniques like **SFT**, **LoRA**, and **QLoRA** feel intuitive later on.

## From RNNs to Attention

Before the rise of Transformers, Recurrent Neural Networks (RNNs) and LSTMs were the primary tools for sequence modeling.

* **The Bottleneck**: RNNs struggled with long sequences because they forced all information through a single, fixed-length compressed state vector.
* **The Solution**: Attention was introduced to allow models to "look back" at different parts of the input dynamically.
* **Mechanism**: It acts as a weighted lookup where a **Query** is compared against **Keys** to determine which **Values** are most relevant.

## Self-Attention: The Key Insight

Self-attention generalizes the attention mechanism so that every token in a sequence attends to every other token in that same sequence.

* **Example**: In the sentence "The cat sat on the mat," the representation for "sat" can attend strongly to "cat" (the actor) and "mat" (the location).
* **Parallelism**: By removing recurrence, self-attention allows for massive parallel processing of data, which is essential for training on modern hardware.

## Architectural Components

* **Positional Encodings**: Since attention is permutation-invariant (it doesn't inherently know word order), Transformers inject order information using sine and cosine functions at different frequencies.
* **Scaled Dot-Product Attention**: This is the mathematical implementation that computes scores between queries and keys, normalizes them with softmax, and weights the values.
* **Multi-Head Attention**: This allows the model to simultaneously focus on different types of information, such as syntax, long-range dependencies, or semantic similarity.

## The Three Transformer Families

Different arrangements of Transformer blocks lead to three main types of models:

1. **Encoder-only (e.g., BERT)**: Best for understanding text, classification, and retrieval through bidirectional self-attention.
2. **Encoder-Decoder (e.g., T5, BART)**: Designed for sequence-to-sequence tasks like translation and summarization.
3. **Decoder-only (e.g., GPT, Llama, Qwen)**: Uses causal self-attention (tokens only see the past) to predict the next token. This architecture powers almost all modern LLMs.

## The Modern LLM Training Pipeline

OpenAI's **InstructGPT** (2022) formalized the standard three-stage pipeline used today:

1. **Pretraining**: Training on massive raw text data using Causal Language Modeling (CLM). This creates a **Base Model** that "learns the world".
2. **Supervised Fine-Tuning (SFT)**: Teaching the model to follow instructions using high-quality example pairs.
3. **Alignment (RLHF)**: Refining the model using human feedback to ensure safety and helpfulness.

## Continued Pretraining

While SFT and LoRA shape existing behavior, **Continued Pretraining** is used to inject new knowledge.

* **Use Cases**: Adding specialized domain knowledge (legal, medical) or supporting an underrepresented language.
* **Lab Preview**: This week’s lab focuses on this stage, extending a base model with a mathematical dataset.

---

## 💡 Review Questions (EN)

**1. Why was attention introduced into sequence models?**

* **Answer**: To solve the "bottleneck" problem of RNNs/LSTMs, where long sentences had to be compressed into a single vector. Attention allows the model to look back at any part of the input as needed.

**2. How does self-attention differ from classic encoder-decoder attention?**

* **Answer**: Encoder-decoder attention links two different sequences (input and output), while self-attention allows tokens within the *same* sequence to interact with each other.

**3. What are the strengths of the three Transformer architectures?**

* **Answer**: Encoder-only is best for understanding/classification; Encoder-decoder is best for seq2seq (translation); Decoder-only is the king of text generation and scaling.

**4. Why isn't a "Base Model" a good conversational assistant?**

* **Answer**: A base model is only trained to complete text. It hasn't learned to follow instructions or adopt a helpful/safe persona; those are added during post-training (SFT/RLHF).

**5. When should you choose Continued Pretraining over SFT?**

* **Answer**: Choose Continued Pretraining when the model lacks foundational knowledge (a new language or highly technical domain data). Use SFT when you just want to change *how* the model responds.

---

---

# 🇫🇷 Résumé en français

## Contexte et Objectifs

Cette première leçon pose les fondations indispensables avant d'aborder le finetuning.

* **Concepts clés** : Compréhension du pré-entraînement, de l'architecture Transformer et de la pipeline complète d'entraînement des LLM modernes.
* **Carte Mentale** : L'objectif est de clarifier la structure avant d'utiliser des techniques comme **SFT**, **LoRA** et **QLoRA**.

## Des RNNs à l'Attention

Avant les Transformers, les modèles RNN et LSTM étaient la norme pour le traitement du langage.

* **Le Goulot d'Étranglement** : Les RNN devaient compresser toute l'information dans un seul vecteur d'état, ce qui nuisait aux performances sur les phrases longues.
* **La Solution** : L'attention permet au modèle de "regarder en arrière" vers n'importe quelle partie de l'entrée de manière dynamique.
* **Mécanisme** : Elle fonctionne comme un système de recherche (lookup) où une requête (**Query**) est comparée à des clés (**Keys**) pour extraire des valeurs (**Values**) pertinentes.

## La Self-Attention : L'Innovation Majeure

La self-attention permet à chaque token d'une séquence de "communiquer" avec tous les autres tokens de cette même séquence.

* **Exemple** : Dans "Le chat est sur le tapis", le mot "est" peut se lier au "chat" (le sujet) et au "tapis" (le lieu).
* **Parallélisme** : En supprimant la récurrence, on peut traiter des séquences entières en même temps, ce qui accélère considérablement l'entraînement.

## Composants de l'Architecture

* **Encodage Positionnel** : Comme l'attention ne connaît pas l'ordre des mots, on injecte cette information via des fonctions sinusoïdales.
* **Scaled Dot-Product Attention** : L'implémentation mathématique qui calcule les scores d'importance entre les mots et normalise les résultats via une fonction softmax.
* **Multi-Head Attention** : Permet au modèle de regarder plusieurs types de relations simultanément (ex: grammaire vs sémantique).

## Les Trois Familles de Transformers

On distingue trois grandes architectures selon l'organisation des blocs :

1. **Encodeur seul (ex: BERT)** : Idéal pour la compréhension, la classification et la recherche sémantique.
2. **Encodeur-Décodeur (ex: T5, BART)** : Optimisé pour les tâches de type "entrée vers sortie" comme la traduction ou le résumé.
3. **Décodeur seul (ex: GPT, Llama, Qwen)** : Prédit le prochain mot. C'est l'architecture qui domine les LLM actuels.

## Pipeline d'Entraînement Moderne

Formalisée par **InstructGPT** en 2022, la pipeline standard comprend trois étapes :

1. **Pré-entraînement** : Apprentissage sur des corpus massifs de texte brut. On obtient un **Modèle de Base** qui "apprend le monde".
2. **Fine-Tuning Supervisé (SFT)** : On apprend au modèle à suivre des instructions via des exemples de haute qualité.
3. **Alignement (RLHF)** : Ajustement via le feedback humain pour garantir l'utilité et la sécurité du modèle.

## Pourquoi Continuer le Pré-entraînement ?

Le finetuning (SFT/LoRA) ne crée pas de nouvelles connaissances, il oriente celles existantes.

* **Cas d'usage** : Pour ajouter un domaine très spécifique (juridique, médical) ou une langue sous-représentée, on utilise le **Pré-entraînement continu**.
* **Lab de la semaine** : Mise en pratique sur un dataset de texte mathématique.

---

## 💡 Questions de révision (FR)

**1. Pourquoi l’attention a-t-elle été introduite dans les modèles séquentiels ?**

* **Réponse** : Pour éviter de compresser toute une phrase dans un seul vecteur (le goulot d'étranglement des RNN). Elle permet de revenir dynamiquement sur n'importe quel mot de l'entrée.

**2. En quoi la self-attention diffère-t-elle de l’attention classique ?**

* **Réponse** : L'attention classique lie une entrée à une sortie. La self-attention permet aux mots d'une même phrase d'interagir entre eux pour mieux comprendre le contexte interne.

**3. Quelles sont les forces des trois architectures ?**

* **Réponse** : L'encodeur pour comprendre, l'encodeur-décodeur pour traduire/résumer, et le décodeur-seul pour la génération massive et le passage à l'échelle.

**4. Pourquoi un modèle de base n'est-il pas un bon assistant conversationnel ?**

* **Réponse** : Parce qu'il cherche seulement à compléter statistiquement le texte. Il n'a pas appris à répondre à une consigne ou à respecter des règles de sécurité.

**5. Quand choisir le pré-entraînement continué plutôt que le SFT ?**

* **Réponse** : On choisit le pré-entraînement continué pour injecter des connaissances brutes massives (ex: nouveau domaine technique). On utilise le SFT pour changer le comportement du modèle.

