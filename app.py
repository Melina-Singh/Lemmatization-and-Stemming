from flask import Flask, request, jsonify, render_template, session
from nlp_processing import NLPProcessor
from logger import setup_logger

# Initialize Flask app for web server and REST API
app = Flask(__name__)
app.secret_key = '12345'  # Required for sessions
logger = setup_logger(__name__)  # Initialize logger for app.py
processor = NLPProcessor()  # Initialize NLP processor

# Route to serve the main web interface
@app.route('/')
def index():
    logger.info("Serving main web interface (index.html)")
    try:
        return app.send_static_file('index.html')
    except Exception as e:
        logger.error(f"Error serving index.html: {str(e)}")
        raise

# REST API endpoint for NLP text processing
@app.route('/api/process', methods=['POST'])
def process_text():
    logger.info("Received API request to process text")
    try:
        data = request.get_json()
        text = data.get('text', '')
        logger.debug(f"Input text: {text}")
        
        if not text.strip():
            logger.warning("Empty input text received")
            return jsonify({"error": "Text input cannot be empty"}), 400
        
        # Store text in session
        session['input_text'] = text
        
        response = {
            'tokens': processor.tokenize(text),
            'lemmas': processor.lemmatize(text),
            'stems': processor.stem(text),
            'pos_tags': processor.pos_tag(text),
            'entities': processor.ner(text),
            'lemma_stem_comparison': processor.compare_lemmas_stems_realtime(text)
        }
        
        logger.info("Successfully processed text")
        logger.debug(f"API response: {response}")
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error processing API request: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Route for lemmatization vs. stemming comparison page
@app.route('/comparison')
def comparison():
    logger.info("Serving lemmatization vs. stemming comparison page")
    try:
        # Get text from query parameter or session
        text = request.args.get('text', session.get('input_text', ''))
        logger.debug(f"Received text for comparison: {text}")
        
        if not text.strip():
            logger.warning("No text provided for comparison, using default text")
            text = "running runs studied"
        
        # Get comparison data using dynamic method
        comparison_data = processor.compare_lemmas_stems_realtime(text)
        logger.debug(f"Comparison data: {comparison_data['comparison']}")
        
        # Preprocess explanation into structured sections
        sections = []
        current_section = None
        lines = comparison_data['explanation'].strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith('Real-Time Lemmatization vs. Stemming Analysis:'):
                sections.append({'title': 'Lemmatization vs Stemming Comparison', 'content': []})
                current_section = sections[-1]
            elif line.startswith('- **') and '**: ' in line:
                title = line.split('**: ')[0][3:].strip()
                sections.append({'title': title, 'content': []})
                current_section = sections[-1]
                current_section['content'].append(line.split('**: ')[1])
            elif current_section:
                current_section['content'].append(line)
        
        logger.debug(f"Parsed sections: {sections}")
        return render_template('comparison.html', 
                             comparisons=comparison_data['comparison'],
                             sections=sections)
    except Exception as e:
        logger.error(f"Error rendering comparison page: {str(e)}")
        raise

if __name__ == '__main__':
    logger.info("Starting Flask server")
    app.run(debug=True)