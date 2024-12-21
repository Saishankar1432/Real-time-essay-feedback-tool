from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
from dotenv import load_dotenv
import logging


load_dotenv()

app = Flask(__name__)


CORS(app, resources={r"/analyze": {"origins": ["http://localhost:3000"]}})  

logging.basicConfig(level=logging.INFO)

openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "Server is running"}), 200

@app.route('/analyze', methods=['POST'])
def analyze_essay():
    try:
        logging.info("Received a request for essay analysis.")
        data = request.get_json()
        essay = data.get('essay')

        if not essay or not isinstance(essay, str) or not essay.strip():
            return jsonify({"error": "Invalid essay. Please provide a non-empty string"}), 400

        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"Analyze this essay for clarity, grammar, tone, and structure. Provide specific suggestions for improvement:\n\n{essay}",
            max_tokens=500,
            temperature=0.7  
        )
        feedback = response.choices[0].text.strip()

        logging.info("Feedback successfully generated.")
        return jsonify({"feedback": feedback})

    except openai.error.AuthenticationError:
        logging.error("Invalid OpenAI API Key.")
        return jsonify({"error": "Invalid OpenAI API Key"}), 401

    except openai.error.RateLimitError:
        logging.error("Rate limit exceeded.")
        return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429

    except openai.error.OpenAIError as e:
        logging.error(f"OpenAI API Error: {str(e)}")
        return jsonify({"error": f"OpenAI API Error: {str(e)}"}), 500

    except Exception as e:
        logging.error(f"Unexpected Error: {str(e)}")
        return jsonify({"error": f"Unexpected Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
