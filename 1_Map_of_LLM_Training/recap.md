# The Finetuning Landscape - A Map of Modern LLM Training

![](https://substackcdn.com/image/fetch/$s_!_Yph!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff1909018-fbb6-4a19-8eb3-7cce5c333837_1920x1080.png)



## English summary

### Context and goals

This first lesson does not dive into finetuning yet; instead, it builds the foundation by explaining **pretraining**, the **Transformer** architecture, and the **modern LLM training pipeline** from pretraining to alignment. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
The goal is to give you a clear mental map so that later techniques like SFT, LoRA, and QLoRA are intuitive rather than magical. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

***

### From RNNs to attention

Before Transformers, **CNNs** dominated vision and **RNNs/LSTMs** were the standard for sequence modeling in translation, speech recognition, and text generation. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
These models struggled with **long sequences** and **long‑range dependencies** because all information had to be funneled through a single compressed state vector. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

**Attention** was first introduced as an improvement to encoder–decoder recurrent models for machine translation. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
Instead of forcing the decoder to rely on one fixed vector, attention lets it dynamically **look back** at different parts of the input at each decoding step. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

At its core, attention is a **weighted lookup mechanism** over keys, values, and a query. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
The query is compared against all keys, scores are turned into weights, and a **weighted sum of the values** gives the attended representation. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

***

### From attention to self‑attention

Initially, attention connects an encoder to a decoder; **self‑attention** generalizes this idea so that **each token in a sequence attends to all other tokens in the same sequence**. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
This removes the need for recurrence and enables fully parallel processing of entire sequences. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

For a sentence like “The cat sat on the mat”, each token gets mapped to **Q, K, V**, then each token’s Q is compared with all Ks, and the resulting distribution is used to mix the Vs. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
The representation of *“sat”* can thus strongly attend to *“cat”* (who is doing the action) and *“mat”* (where it happens), while mostly ignoring function words like *“the”.* [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

Self‑attention is the key insight that makes it possible to build Transformers **without RNNs**. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
Stacking such layers gives a powerful architecture that can model complex dependencies across tokens. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

***

### Positional encodings, scaled dot‑product, and multi‑head

Pure attention is permutation‑invariant: it has **no notion of order** by itself. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
Transformers inject order information via **positional encodings** added to token embeddings. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

The original Transformer uses **sinusoidal positional encodings**, built from sine and cosine functions at different frequencies. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
Low‑frequency components capture coarse position, while high‑frequency components capture fine‑grained differences, together forming a rich position signal. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

The concrete implementation of attention is **scaled dot‑product attention**, which computes dot products between queries and keys, scales them, applies a **softmax**, and uses the result to weight the values. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
Scaling by the square root of the key dimension stabilizes training as dimensionality grows. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

With only one head, the model must compress all relations into a single view. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
**Multi‑head attention** runs several attention mechanisms in parallel on different representation subspaces, letting each head specialize in different patterns such as syntax, locality, or semantic similarity. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

***

### The three Transformer architectures

The same Transformer building blocks can be arranged into three major architectures. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

1. **Encoder‑only**  
   - Takes an input sequence and outputs contextualized token representations via bidirectional self‑attention. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
   - Used for **understanding** tasks (classification, retrieval, semantic similarity), not for text generation. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
   - Example: **BERT**, trained with masked language modeling. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

2. **Encoder–decoder**  
   - The encoder reads the input sequence, the decoder generates the output sequence token by token. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
   - The decoder uses **cross‑attention** over encoder states plus **causal self‑attention** to avoid looking into the future. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
   - Well suited for sequence‑to‑sequence tasks like translation and summarization (e.g., T5, BART). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

3. **Decoder‑only**  
   - Drops the encoder and relies on a single stack of **causal self‑attention** layers. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
   - Trained with a simple **next‑token prediction** objective (causal language modeling). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
   - This is the architecture behind almost all modern LLMs like GPT‑style models, ChatGPT, Claude, and Qwen. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

The bootcamp focuses on decoder‑only models, since they underpin today’s most capable general‑purpose language models. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

***

### The modern LLM training pipeline

In 2022, **InstructGPT** clearly formalized the multi‑stage training pipeline that has become standard for LLMs. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
Earlier models were mostly next‑token predictors; InstructGPT added stages dedicated to following instructions and aligning with human preferences. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

The now‑classic **three‑stage view** is: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

1. **Pretraining** on large‑scale raw text data. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
2. **Supervised Fine‑Tuning (SFT)** on high‑quality, task‑oriented instruction–response examples. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
3. **Alignment via RLHF**, using human feedback to train a reward model and optimize behavior. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

A broader **two‑phase view** is also common: **pretraining** (learning general capabilities) vs **post‑training** (refining, adapting, and aligning those capabilities). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
Decoder‑only Transformers, with their simple objective and compatibility with massive unlabeled corpora, scale particularly well in this framework. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

Empirical work shows **power‑law scaling**: performance improves smoothly with more parameters, more data, and more compute. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
Recent breakthroughs are largely due to **scaling Transformers** rather than inventing completely new architectures. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

***

### Pretraining and base models

Pretraining a decoder‑only Transformer means training it to **predict the next token** given all previous tokens, i.e., **causal language modeling**. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
There are no explicit instructions, labels, or human annotations—just raw text. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

This is **self‑supervised learning**: the next token acts as the supervision signal. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
Because the web and other sources provide virtually unlimited raw text (web pages, books, articles, code repositories), pretraining is highly scalable. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

During this phase, the model learns grammar, style, structure, factual knowledge, code patterns, and possibly multiple languages if present in the data. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
This is why pretraining is often described as the stage where the model **“learns the world.”** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

The result is a **base model** (or foundation model) that is excellent at **continuing text** but not yet reliable at following instructions, being safe, or behaving like a helpful assistant. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
The **Shoggoth meme** captures this: a powerful pattern learner with unrefined, unpredictable behavior. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

***

### Why and when to use continued pretraining

The techniques covered later—SFT, LoRA, QLoRA—do **not** teach language from scratch. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
Instead, they **shape and steer** the capabilities acquired during pretraining into usable behaviors for specific tasks and products. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

However, there are real‑world scenarios where **continued pretraining** is appropriate: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

- When you want to inject **new domain knowledge**, such as legal, medical, or highly technical text. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
- When you need to support an **underrepresented language**. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
- When your data distribution is very different from what the original model saw. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

In these cases, you extend an existing base model by pretraining it further on new data, letting it absorb new patterns while retaining its previous knowledge. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
This is exactly what the accompanying lab will walk through, using a dataset rich in mathematical text. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

***

## Review questions (EN)

1. **Why was attention introduced into sequence models, and which limitation of RNNs/LSTMs does it address?**  
   **Answer:** Attention was introduced to relieve the bottleneck of encoding an entire source sequence into a single state vector, which is problematic for long sentences. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
   It lets the decoder **dynamically focus** on different parts of the input at each step, improving handling of long‑range dependencies and boosting tasks like translation. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

2. **How does self‑attention differ from classic encoder–decoder attention?**  
   **Answer:** In encoder–decoder attention, one sequence (the decoder) attends to another (the encoder). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
   In **self‑attention**, each token attends to all tokens in **the same sequence**, allowing the model to capture internal relationships without recurrence and to process the entire sequence in parallel. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

3. **What are the strengths of encoder‑only, encoder–decoder, and decoder‑only architectures respectively?**  
   **Answer:**  
   - Encoder‑only models shine at **understanding** tasks (classification, retrieval, similarity) via bidirectional context but do not generate text. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
   - Encoder–decoder models are ideal for **sequence‑to‑sequence** tasks where both input and output are sequences of possibly different lengths (e.g., translation, summarization). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
   - Decoder‑only models specialize in **next‑token prediction** and, when scaled up, support **in‑context learning** and general‑purpose text generation, making them central to modern LLMs. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

4. **What exactly does pretraining produce, and why is that not yet a good conversational assistant?**  
   **Answer:** Pretraining yields a **base model** that is extremely good at continuing text and capturing linguistic and factual regularities. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
   However, it has not learned to **follow instructions**, enforce safety, or adopt a user‑friendly conversational style; those behaviors are added during SFT and RLHF in the post‑training phase. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

5. **In which practical cases would you prefer continued pretraining over just doing SFT (e.g., with LoRA) on a base model?**  
   **Answer:** SFT (including parameter‑efficient variants) is suitable when the model already knows the **domain** and **language**, and you mainly want to adjust behavior and task performance. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
   If the domain or language is **severely underrepresented** in the original data, or the distribution is very different (e.g., highly specialized legal corpora), **continued pretraining** is more appropriate to genuinely enrich the model’s internal knowledge before finetuning it. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)



***



## Résumé en français

### Contexte et objectifs

Cette première leçon ne parle pas encore de *finetuning* au sens strict, mais pose les fondations : comprendre le **pré‑entraînement**, l’architecture **Transformer** et la **pipeline complète d’entraînement des LLM modernes**, de la pré‑formation jusqu’à l’alignement. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
L’idée est de construire une carte mentale claire avant d’aborder SFT, LoRA, QLoRA et autres techniques de post‑entraînement. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

```markdown
![](https://substack-post-media.s3.amazonaws.com/public/images/a13c886e-05c9-4da4-aeec-3e4c6deeb046_1200x630.png)
```

Cette vue d’ensemble explique pourquoi les Transformers ont remplacé les RNN/LSTM dans le NLP, et comment on passe d’un **modèle de base** (pré‑entraîné) à un **assistant aligné** de type ChatGPT. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

***

### Des RNNs à l’attention

Avant les Transformers, les **CNN** dominaient la vision, tandis que les **RNN / LSTM** étaient la référence pour la traduction, la reconnaissance vocale et la génération de texte. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
Ces modèles séquentiels souffraient de difficultés sur les **longues séquences** et les **dépendances à long terme**, car tout devait transiter par un vecteur d’état compressé. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

L’**attention** apparaît d’abord comme un *add‑on* aux modèles encodeur‑décodeur récurrents pour la traduction automatique. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
Plutôt que de compresser toute la phrase source dans un seul vecteur, le décodeur apprend à **“regarder en arrière”** différentes positions de l’entrée à chaque étape de génération. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

Conceptuellement, l’attention est un **mécanisme de lookup pondéré** : on a des **keys** (k₁…kₙ), des **values** (v₁…vₙ) et un **query** qui exprime ce que l’on cherche. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
Le query est comparé à toutes les keys, on obtient des poids, puis on calcule une **somme pondérée des values** — c’est ce qu’on appelle souvent *attention pooling*. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

> Intuition : l’attention décide quelles informations comptent le plus pour la représentation courante et les combine de façon différentiable et parallélisable. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

***

### De l’attention à la *self‑attention*

La première étape : l’attention sert de pont entre un **encodeur** (entrée) et un **décodeur** (sortie). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
La généralisation clé est la **self‑attention** : chaque token d’une séquence peut assister à **tous les autres tokens de la même séquence** (y compris lui‑même). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

Pour une phrase comme :

```text
The cat sat on the mat
```

chaque token est projeté en **Q (query)**, **K (key)** et **V (value)**, puis pour chaque token on compare son Q à tous les K, on normalise les scores et on agrège les V correspondants. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
Ainsi, la représentation de *“sat”* peut fortement assister à *“cat”* (qui fait l’action) et à *“mat”* (où elle se déroule), tout en ignorant en grande partie des mots fonctionnels comme *“the”.* [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

La self‑attention, en supprimant la dépendance à la récurrence, permet un **traitement massivement parallèle** des séquences. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
Cela ouvre la voie à l’architecture Transformer : plus besoin de RNN, l’ensemble de la séquence est traitée via des couches d’attention auto‑appliquée. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

***

### Encodage positionnel, attention à produit scalaire et multi‑têtes

L’attention “pure” ne connaît pas l’ordre des tokens : la permutation de la séquence ne change pas les scores si on ne fait rien de plus. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
Les Transformers injectent donc l’information d’ordre via des **encodages positionnels**, qui sont ajoutés aux embeddings de tokens. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

Dans le Transformer d’origine, ces encodages sont **sinusoïdaux** (sin et cos à différentes fréquences). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
Les basses fréquences capturent la position globale, les hautes fréquences affinent les différences locales, ce qui donne une représentation riche des positions. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

L’implémentation concrète de l’attention est la **scaled dot‑product attention** : on calcule les produits scalaires Q·Kᵀ, on les normalise avec une **softmax**, puis on les applique aux V pour obtenir la nouvelle représentation. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
La mise à l’échelle (division par √dₖ) stabilise les gradients quand la dimension des clés augmente. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

Avec une seule tête, le modèle doit tout résumer dans une unique “vue” de la séquence. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
La **multi‑head attention** duplique ce mécanisme sur plusieurs sous‑espaces de représentation, ce qui permet à différentes têtes de se spécialiser en patrons syntaxiques, dépendances locales/longues ou similarité sémantique. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

***

### Les trois architectures Transformer

Le même bloc Transformer peut être organisé en trois grandes familles d’architectures, chacune adaptée à un type de tâche. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

1. **Encodeur seul (encoder‑only)**  
   - Prend une séquence en entrée et produit des représentations riches par token (self‑attention bidirectionnelle). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
   - Pas de génération de texte, uniquement **encodage** pour des tâches comme classification, recherche, similarité sémantique. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
   - Exemple : **BERT**, pré‑entraîné via *masked language modeling* (masking partiel de tokens). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

2. **Encodeur–décodeur (seq2seq)**  
   - L’encodeur lit l’entrée, le décodeur génère la sortie token par token. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
   - Utilise la **cross‑attention** (le décodeur regarde les sorties de l’encodeur) et une self‑attention **causale** côté décodeur (pas le droit de voir le futur). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
   - Idéal pour traduction, résumé, text‑to‑text général (T5, BART). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

3. **Décodeur seul (decoder‑only)**  
   - Supprime l’encodeur et ne garde qu’un décodeur en **self‑attention causale** sur une séquence croissante. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
   - Objectif simple : **prédire le prochain token** à partir de tous les précédents (causal LM). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
   - C’est l’architecture de quasiment tous les LLM modernes : GPT‑like, ChatGPT, Claude, Qwen, etc. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

Cette dernière famille est celle sur laquelle le bootcamp se concentre, car elle sous‑tend les assistants de conversation et les modèles de génération généralistes actuels. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

***

### Pipeline d’entraînement des LLM modernes

```markdown
![](https://substack-post-media.s3.amazonaws.com/public/images/48baf6dd-9664-4ccc-9ebf-4fca2bb400fe_1200x394.png)
```

En 2022, **InstructGPT** d’OpenAI a cristallisé la pipeline moderne des LLM : un **processus en plusieurs étapes** qui est devenu le standard de facto. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
Avant cela, la plupart des modèles étaient uniquement des prédicteurs de prochain token, bons pour compléter du texte, mais peu adaptés au suivi d’instructions ou à l’alignement sur les préférences humaines. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

La vue “historique” en **trois étapes** : [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

1. **Pré‑entraînement** sur un corpus massif de texte brut (objectif CLM). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
2. **Supervised Fine‑Tuning (SFT)** sur des exemples d’instructions et de réponses de haute qualité. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
3. **Alignement via RLHF** (Reinforcement Learning from Human Feedback) pour optimiser l’utilité et la sécurité. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

On parle aussi souvent d’une vue en **deux phases** : [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

- **Pré‑entraînement** : acquisition des capacités générales de langage.  
- **Post‑training** (SFT, RLHF, variantes) : adaptation, spécialisation, alignement du comportement. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

Les Transformers, et en particulier les décodeurs‑seuls, se prêtent extrêmement bien au **scaling** : performance qui croît de manière régulière avec la taille du modèle, la taille du corpus et le compute. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
Les grandes avancées récentes viennent moins de nouvelles architectures radicales que du **changement d’échelle** sur cette pipeline. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

***

### Pré‑entraînement et modèles de base

```markdown
![](https://substack-post-media.s3.amazonaws.com/public/images/a2a095ff-c348-4a7a-a09f-5228019d8246_3106x1888.png)
```

Le **pré‑entraînement** d’un décodeur‑seul consiste à lui faire prédire le **prochain token** dans d’énormes corpus de texte, en mode **causal language modeling (CLM)**. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
Pas d’instructions, pas d’annotations humaines, pas d’étiquettes de tâche : uniquement du texte brut, token après token. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

C’est une forme de **self‑supervised learning** : le label est dans les données elles‑mêmes, le prochain token jouant le rôle de cible. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
Cela rend le processus hautement **scalable**, puisque quasiment tout texte tokenisable (web, livres, articles, dépôts de code) peut servir de données. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

Pendant cette phase, le modèle apprend : grammaire, style, structures de texte, faits du monde, patterns de code, éventuellement multilingue si les données le sont. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
On décrit souvent le pré‑entraînement comme le moment où le modèle **“apprend le monde”**. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

Le résultat est un **modèle de base** (*base model* ou *foundation model*) : très bon pour **continuer du texte**, mais pas encore fiable pour suivre des consignes, refuser des requêtes dangereuses ou adopter un style conversationnel cohérent. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
Le “mème du Shoggoth” (monstre brut recouvert d’un masque souriant) illustre bien ce stade : énorme capacité de pattern‑matching, mais comportement non aligné. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

***

### Pourquoi et quand continuer le pré‑entraînement

Les techniques dont traitera le bootcamp (SFT, LoRA, QLoRA, etc.) **n’apprennent pas le langage à partir de zéro**. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
Elles **façonnent** et **orientent** les capacités déjà acquises pendant le pré‑entraînement pour les adapter à des usages concrets. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

Il existe cependant des cas où il est pertinent de faire du **pré‑entraînement continué** (*continued pretraining*) plutôt que (ou en plus de) du simple finetuning : [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

- Ajouter de la **connaissance de domaine** (juridique, médicale, technique, etc.). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
- Couvrir une **langue sous‑représentée** dans les données initiales. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
- Adapter le modèle à une **distribution de données très différente** de celle de départ. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

Dans ces scénarios, on repart d’un modèle de base et on poursuit le pré‑entraînement sur un nouveau corpus, pour lui faire absorber de nouveaux patterns tout en conservant ce qu’il sait déjà. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
C’est précisément ce qui sera mis en pratique dans le lab de la semaine, avec un dataset contenant du texte mathématique. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

***

## Questions de révision (FR)

1. **Pourquoi l’attention a‑t‑elle été introduite dans les modèles séquentiels, et quel problème des RNN/LSTM cherche‑t‑elle à résoudre ?**  
   **Réponse :** L’attention a été introduite dans les modèles encodeur‑décodeur pour éviter de compresser toute la séquence source dans un seul vecteur d’état, ce qui devient limitant pour les longues phrases. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
   Elle permet au décodeur de **revenir dynamiquement** sur différentes positions de l’entrée, améliorant la gestion des dépendances longues et la qualité de tâches comme la traduction. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

2. **En quoi la self‑attention diffère‑t‑elle de l’attention classique encodeur–décodeur ?**  
   **Réponse :** Dans l’attention encodeur–décodeur, une séquence (décodeur) assiste à une autre séquence (encodeur). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
   Dans la **self‑attention**, chaque token assiste à tous les tokens de **la même séquence**, ce qui permet de modéliser les relations internes à la phrase sans récurrence et de tout paralléliser. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

3. **Quelles sont les forces respectives des architectures encodeur‑seul, encodeur–décodeur et décodeur‑seul ?**  
   **Réponse :**  
   - L’encodeur‑seul excelle pour **comprendre** un texte (classification, similarité, retrieval) via une attention bidirectionnelle, mais ne génère pas de texte. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
   - L’encodeur–décodeur est idéal pour les tâches **séquence‑à‑séquence** (traduction, résumé) où entrée et sortie sont deux séquences potentiellement de longueur différente. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
   - Le décodeur‑seul est spécialisé dans la **prédiction du prochain token** et, à grande échelle, permet l’**in‑context learning** et la génération générale de texte, ce qui le rend central pour les LLM modernes. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

4. **Que produit exactement la phase de pré‑entraînement, et pourquoi un modèle ainsi obtenu n’est‑il pas encore un bon assistant conversationnel ?**  
   **Réponse :** Le pré‑entraînement produit un **modèle de base** très performant pour compléter du texte et capturer des régularités linguistiques et factuelles. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
   Cependant, il n’a pas appris à **suivre des instructions**, à gérer la sécurité ou à adopter un style de réponse adapté à l’humain ; ces comportements sont ajoutés via SFT et RLHF pendant le post‑training. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

5. **Dans quels cas pratiques choisirais‑tu un pré‑entraînement continué plutôt qu’un simple SFT sur un modèle de base (par exemple via LoRA) ?**  
   **Réponse :** Le SFT (même param‑efficient) est approprié si le modèle connaît déjà le **domaine** et la **langue**, et que l’on veut surtout modifier le comportement (format, style, alignement sur des tâches précises). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)
   En revanche, si le domaine ou la langue sont **très sous‑représentés** dans les données initiales, ou si la distribution est radicalement différente (ex. corpus juridique très spécialisé), un **pré‑entraînement continué** est mieux adapté pour enrichir réellement les connaissances internes du modèle avant de le finetuner. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/157922223/beb966d3-42cd-44db-b322-dd80dca47fd9/1_The-Finetuning-Landscape-A-Map-of-Modern-LLM-Training.md)

***

