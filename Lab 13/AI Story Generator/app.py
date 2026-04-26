import os
import time
from flask import Flask, render_template, request
from google import genai 
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
hf_client = InferenceClient(token=os.getenv("HF_TOKEN"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    user_prompt = request.form.get('prompt')

    try:
        # --- PART A: GENERATE THE STORY ---
        # Using 2.5-flash: The most stable model 
        story_response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=f"Write a professional 3-paragraph story based on: {user_prompt}"
        )
        story_text = story_response.text

        # --- PART B: GENERATE THE IMAGE ---
        image_prompt = f"Storybook illustration of {user_prompt}, vibrant colors, digital art style, high resolution"
        
        # Hugging Face models don't change as often, this remains stable
        image = hf_client.text_to_image(
            image_prompt, 
            model="black-forest-labs/FLUX.1-schnell"
        )

        # Ensure the path is correct for Windows
        image_filename = "generated_story.png"
        image_save_path = os.path.join("static", "images", image_filename)
        image.save(image_save_path)

        return render_template(
            'story.html', 
            story_text=story_text, 
            image_url=image_save_path
        )

    except Exception as e:
        # If it's still a quota error, we provide a clear instruction
        if "429" in str(e):
            return "<h3>Server Busy: Please wait 30 seconds and click 'Generate' again.</h3>"
        return f"System Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)