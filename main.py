import os
from flask import Flask, request, render_template, jsonify
import json
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# --- New: Load environment variables (for the API key) ---
load_dotenv()

# --- New: Initialize the Gemini LLM ---
# The library automatically finds your GOOGLE_API_KEY from the .env file
# Changed model from "gemini-pro" to a newer, available model "gemini-1.5-flash"
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# Initialize the Flask app
app = Flask(__name__)

# Load the knowledge base from a JSON file
try:
    with open("knowledge_base.json", "r", encoding="utf-8") as f:
        knowledge_base = json.load(f)
except FileNotFoundError:
    print("Error: knowledge_base.json not found. Please create one.")
    knowledge_base = {} # Start with an empty knowledge base if file is missing

# Cache for frequently asked questions to reduce API calls
response_cache = {}

# --- Changed: The function is now simpler and uses Gemini ---
def generate_response_with_gemini(user_query, language="en"):
    """Generate a response using Gemini, referencing the knowledge base."""
    # Check if the response is already cached
    cache_key = f"{user_query}_{language}"
    if cache_key in response_cache:
        return response_cache[cache_key]

    # Create a detailed prompt that includes the knowledge base context
    prompt = f"""
    You are a helpful University support chatbot. Your task is to answer user queries based ONLY on the following knowledge base.
    If the answer is not in the knowledge base, politely say that you don't have that information. Do not make up answers.

    Knowledge Base:
    {json.dumps(knowledge_base, indent=2)}

    User Query: {user_query}
    """

    # --- Changed: Simplified API call using LangChain ---
    # The complex async/await logic is no longer needed.
    try:
        response_content = llm.invoke(prompt).content
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        response_content = "Sorry, I'm having trouble connecting to my brain right now. Please try again later."

    # --- MODIFIED SECTION ---
    # If the selected language is not English, translate the response.
    if language != "en":
        try:
            # Use deep-translator to convert the response to the target language.
            # The source language is detected automatically.
            response_content = GoogleTranslator(source="auto", target=language).translate(response_content)
        except Exception as e:
            print(f"Error during translation to '{language}': {e}")
            # Fallback to the original (likely English) response if translation fails
            pass

    # Cache the final response and return it
    response_cache[cache_key] = response_content
    return response_content

# --- Flask routes remain the same ---

@app.route("/")
def home():
    """Render the main chat page."""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """Handle the chat request from the user."""
    user_query = request.form.get("user_query")
    language = request.form.get("language", "en")  # Default to English

    if not user_query:
        return jsonify({"response": "Please enter a question."})

    # Generate a response using the new Gemini function
    response = generate_response_with_gemini(user_query, language)

    return jsonify({"response": response})


if __name__ == "__main__":
    # Ensure you have your index.html and knowledge_base.json in the correct folders.
    app.run(debug=True)