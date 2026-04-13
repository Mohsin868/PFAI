# Lab 11: Introduction to Generative AI & Retrieval Systems

## Task 1: Conceptual Comparison and Definitions

This task focuses on the core components of modern AI pipelines, specifically distinguishing between generation engines, frameworks, and specialized storage systems.

---

### 1. Generative AI (GenAI) vs. GANs
* **Generative AI (GenAI):** The broad umbrella term for any AI capable of creating new content (text, images, code, audio) rather than just classifying or analyzing existing data.
* **GANs (Generative Adversarial Networks):** A specific architecture within GenAI consisting of two neural networks—a **Generator** (creating fake data) and a **Discriminator** (trying to spot the fake)—competing against each other. While historically dominant for image generation, they have largely been superseded by Diffusion models.

### 2. LLMs (Large Language Models) vs. LangChain
* **LLMs:** The "brain" of the operation. These are massive neural networks trained on vast amounts of text to predict the next token in a sequence, providing raw reasoning and language capabilities.
* **LangChain:** The "framework" or orchestration layer. It is a library used to "chain" LLMs with external tools (APIs, databases, or local files). It allows the LLM to interact with the real world beyond its static training data.

### 3. Vectors vs. VectorDB
* **Vector:** A mathematical representation of the "meaning" of data. In AI, text or images are converted into long arrays of numbers (embeddings). Data points with similar meanings are positioned closer together in this mathematical space.
* **VectorDB:** A specialized database (e.g., Pinecone, Milvus) built to store and search these vectors. Unlike traditional SQL databases that look for exact keyword matches, a VectorDB searches for **semantic similarity**.

### 4. FAISS (Facebook AI Similarity Search)
* **FAISS:** Developed by Meta, this is a highly efficient library for dense vector similarity search. While a VectorDB provides a full management system, FAISS is the core engine that enables searching through millions of vectors in milliseconds to find "nearest neighbors."

### 5. RAG (Retrieval-Augmented Generation)
**RAG** is the architectural pattern used to give an LLM "open-book" access to private or real-time data. The process follows these steps:
1.  **Query:** A user asks a question.
2.  **Retrieval:** The system searches a **VectorDB** (often using **FAISS**) to find relevant context.
3.  **Augmentation:** The retrieved context is added to the user's original prompt.
4.  **Generation:** The **LLM** generates a response based *only* on the provided context, significantly reducing hallucinations.

---

## Technical Summary Table

| Concept | Primary Function | Analogy |
| :--- | :--- | :--- |
| **LLM** | Reasoning & Processing | The Student |
| **LangChain** | Workflow & Automation | The Student's Desk & Tools |
| **Vector** | Semantic Representation | Coordinates on a map |
| **VectorDB** | Long-term Memory | The Library |
| **RAG** | Contextual Accuracy | An Open-book Exam |