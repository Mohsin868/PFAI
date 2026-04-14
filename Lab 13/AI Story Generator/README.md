# Visual AI Storybook Architect 🎨📚

**Author:** Mohsin Ahmad


A cutting-edge Multi-Modal Generative AI application that transforms a single creative prompt into a structured 3-paragraph narrative and a high-fidelity digital illustration. This project demonstrates the orchestration of two distinct state-of-the-art AI models via a Flask backend.

---

## 🚀 Core Tech Stack

- **Backend Framework:** Flask (Python 3.12+)
- **LLM (Text Generation):** Google Gemini 2.5 Flash (via `google-genai`)
- **Image Generation:** FLUX.1-schnell (via Hugging Face Inference API)
- **Environment Management:** Python Dotenv & Virtual Environments

---

## 🧠 System Architecture

The application follows an **Agentic Workflow** pattern:
1. **User Input:** Receives a creative theme or prompt via a web form.
2. **Narrative Synthesis:** The prompt is sent to **Gemini 2.5 Flash**, which acts as the "Story Architect" to generate a coherent 3-paragraph story.
3. **Visual Contextualization:** A specialized image prompt is dynamically generated from the user's input and sent to the **FLUX.1 Diffusion Model**.
4. **Media Management:** The generated image is saved locally to the `static/images/` directory to ensure fast loading and persistent storage.
5. **Output Rendering:** The final multi-modal content is served via a Jinja2-powered frontend.

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
 https://github.com/Mohsin868/AI-Story-Generator-for-Kids/tree/main 

### 2. Set up Virtual Environment
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

### 3.Install Dependencies

pip install -r requirements.txt

### 4. Environment Variables

Create a .env file in the root directory:

GEMINI_API_KEY=your_google_ai_studio_key

HF_TOKEN=your_huggingface_token