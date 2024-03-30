from flask import Flask, jsonify, request
from flask_cors import CORS
import PyPDF2
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)
CORS(app)

model_path = './pegasus-new'
tokenizer_path = './tokenizer'

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(
    model_path, use_safetensors=True)


def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page_number in range(len(pdf_reader.pages)):
            page_text = pdf_reader.pages[page_number].extract_text()
            page_text = page_text.replace('-', '')
            page_text = ' '.join(page_text.split())
            text += page_text + ' '
        return text.strip()


@app.route('/api/home', methods=['GET'])
def return_home():
    return jsonify({
        'message': 'hello world'
    })


@app.route('/api/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')

    if file and file.filename.endswith('.pdf'):
        file_path = 'uploads/' + file.filename
        file.save(file_path)

        text = extract_text_from_pdf(file_path)

        return jsonify({'message': 'Text extracted successfully', 'text': text}), 200
    elif not file:
        return jsonify({'error': 'No file provided'}), 404
    else:
        return jsonify({'error': 'Invalid file format. Only PDF files are supported.'}), 403


@app.route('/api/summary', methods=['POST'])
def get_summary():
    input_text = request.form.get('inputText')
    max_len = request.form.get('length')
    max_len = int(max_len)
    print("given length is:: ", max_len, type(max_len))
    if input_text:
        temp = tokenizer(input_text, max_length=1024,
                         truncation=True, return_tensors="pt")
        output = model_pegasus.generate(input_ids=temp["input_ids"],
                                        attention_mask=temp["attention_mask"], min_length=max_len, max_length=1000)
        decode = [tokenizer.decode(
            s, skip_special_tokens=True, clean_up_tokenization_spaces=True) for s in output]

        return jsonify({'message': 'Text received successfully', 'summary': decode[0]}), 200
    else:
        return jsonify({'error': 'No input text provided'}), 404


if __name__ == "__main__":
    app.run(debug=True, port=8080)


# def extract_text_from_pdf(file_path):
#     with open(file_path, 'rb') as pdf_file:
#         pdf_reader = PyPDF2.PdfReader(pdf_file)
#         text = ''
#         for page_number in range(len(pdf_reader.pages)):
#             page = pdf_reader.pages[page_number]
#             lines = page.extract_text().split('\n')
#             for i in range(len(lines) - 1):
#                 if lines[i].strip().endswith('-') and lines[i + 1]:
#                     lines[i] = lines[i][:-1] + lines[i + 1].strip()
#                     lines[i + 1] = ''
#             text += '\n'.join(lines) + '\n'
#         return text
