from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app = Flask(__name__)

# ---------------------------------------------------------
# 1. PREPROCESSING & DATASET
# ---------------------------------------------------------
# Text-based QnA data from Lab 10
qa_dataset = [
    {"q": "Where is the hotel located?", "a": "We are located in DHA Phase 5, Lahore, near Main Boulevard."},
    {"q": "What are the room categories and prices?", "a": "We offer Standard ($100), Deluxe ($150), and Suites ($250)."},
    {"q": "What food is available in the restaurant?", "a": "Our menu includes Pakistani (Biryani), Chinese (Manchurian), and Continental (Steaks)."},
    {"q": "What facilities do you provide?", "a": "Amenities include Free WiFi, Swimming Pool, Gym, and 24/7 Room Service."},
    {"q": "How do I make a reservation?", "a": "You can book a room by telling me the room type and your check-in date."},
    {"q": "Is parking available?", "a": "Yes, we provide free secure parking for all our guests."}
]

questions = [item['q'] for item in qa_dataset]
answers = [item['a'] for item in qa_dataset]

# ---------------------------------------------------------
# 2. EMBEDDING (Hugging Face MiniLM)
# ---------------------------------------------------------
# Loading the model to convert text into numerical vectors
model = SentenceTransformer('all-MiniLM-L6-v2')
question_embeddings = model.encode(questions)

# ---------------------------------------------------------
# 3. VECTOR STORAGE (FAISS)
# ---------------------------------------------------------
# Create the FAISS index and add our question vectors
dimension = question_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(question_embeddings).astype('float32'))

# ---------------------------------------------------------
# 4. SEARCH & MATCH LOGIC
# ---------------------------------------------------------
def get_best_match(user_query):
    # Convert user query to vector
    query_vec = model.encode([user_query])
    
    # Search for the 1 nearest neighbor
    distances, indices = index.search(np.array(query_vec).astype('float32'), k=1)
    
    # Distance threshold (if > 1.5, the question is likely unrelated)
    if distances[0][0] < 1.5:
        return answers[indices[0][0]]
    else:
        return "I'm sorry, I don't have information on that specific topic. Try asking about our rooms, location, or food."

# ---------------------------------------------------------
# 5. FLASK ROUTES
# ---------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot():
    user_data = request.get_json()
    user_input = user_data.get("message")
    
    if not user_input:
        return jsonify({"reply": "Please enter a message."})
        
    response = get_best_match(user_input)
    return jsonify({"reply": response})

if __name__ == "__main__":
    app.run(debug=True)