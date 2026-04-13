# Lab 11 - Tasks

## Task 1: Describe the Difference Between the Following Concepts

### 1. Generative AI (GenAI) vs. GANs
* **Generative AI (GenAI):** This is the broad umbrella term for any AI capable of creating new content (text, images, code, audio) rather than just classifying existing data.
* **GANs (Generative Adversarial Networks):** A specific architecture within GenAI. It consists of two neural networks. A Generator creating fake data and a Discriminator trying to spot the fake competing against each other. GANs were the standard for image generation before Diffusion models became popular.

### 2. LLMs (Large Language Models) vs. LangChain
* **LLMs:** The "engine" or the brain. These are massive neural networks (like GPT-4 or Llama-3) trained on text to predict the next token in a sequence. They provide the raw intelligence.
* **LangChain:** The "framework" or the glue. LLMs are powerful but "forgetful" and can't easily browse the web or read your local files. LangChain is a library that allows you to chain an LLM with other tools (like your car dataset or a calculator) to create a functional application.

### 3. Vectors vs. VectorDB
* **Vector:** In AI, a vector is a long list of numbers (an array) that represents the "meaning" of a piece of data. If you convert the word "Car" into a vector, it might look like `[0.12, -0.54, 0.88...]`.
* **VectorDB:** A specialized database designed to store and search these lists of numbers. Unlike a traditional SQL database that searches for exact keywords, a VectorDB searches for semantic similarity (finding things that mean the same thing, even if the words are different).

### 4. FAISS (Facebook AI Similarity Search)
* **FAISS:** This is a specific library developed by Meta. While a VectorDB is the "container," FAISS is the "engine" inside that allows you to search through millions of vectors in milliseconds. It is incredibly efficient at finding the "nearest neighbor" for a search query.

### 5. RAG (Retrieval-Augmented Generation)
**RAG** is the process of giving an LLM "open-book" access to your data:
1. A user asks a question.
2. The system searches a VectorDB (using FAISS) for relevant documents.
3. The system feeds those documents + the question to the LLM.
4. The LLM generates an answer based on your specific data, reducing hallucinations.