from flask import Flask, request, jsonify
from pdfminer.high_level import extract_text
from io import BytesIO

app = Flask(__name__)

@app.route('/read_pdf', methods=['POST'])
def read_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        file_stream = BytesIO(file.read())
        text = extract_text(file_stream)
        print("Extracted text:", text)  # Print the extracted text to the terminal
        return jsonify({'text': text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)  # Enable debug mode for better error messages
