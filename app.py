from flask import Flask, request, jsonify
import json
import boto3
import pandas as pd
import requests
from io import BytesIO

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

# Initialize Bedrock client
bedrock = boto3.client('bedrock-runtime')

def process_product(product_name, product_description):
    prompt_data = (
        f"Provide product details for '{product_name}' in the following JSON format:\n"
        "{\n"
        f"    \"Product Name\": \"{product_name}\",\n"
        f"    \"Dimensions\": <width> x <height> x <length>,\n"
        f"    \"Perishable\": \"<True/False>\",\n"
        f"    \"Explosive\": \"<True/False>\"\n"
        "}\n"
        "The dimensions should be randomized numbers in inches, and the perishable and explosive properties should be based on one of the words in the product name."
    )
    payload = {
        "inputText": prompt_data,
        "textGenerationConfig": {
            "maxTokenCount": 3072,
            "stopSequences": [],
            "temperature": 0.7,
            "topP": 0.9
        }
    }

    body = json.dumps(payload)
    model_id = "amazon.titan-text-express-v1"
    
    try:
        response = bedrock.invoke_model(
            body=body,
            modelId=model_id,
            accept="application/json",
            contentType="application/json",
        )
        
        response_text = response['body'].read().decode('utf-8')
        response_json = json.loads(response_text)
        
        output_text = response_json['results'][0]['outputText']
        
        dimensions, perishable, explosive = "Not provided", "Not provided", "Not provided"
        
        lines = output_text.split('\n')
        for line in lines:
            if 'dimensions' in line.lower():
                dimensions = line.split(':', 1)[1].strip().strip(',')
            elif 'perishable' in line.lower():
                perishable = 'True' if 'true' in line.lower() else 'False'
            elif 'explosive' in line.lower():
                explosive = 'True' if 'true' in line.lower() else 'False'
        
    except (json.JSONDecodeError, KeyError):
        pass
    
    return dimensions, perishable, explosive

@app.route('/process', methods=['POST'])
def process_file():
    try:
        data = request.json
        file_url = data['fileUrl']

        response = requests.get(file_url)
        file_data = response.content

        df = pd.read_excel(BytesIO(file_data))

        results = []
        for index, row in df.iterrows():
            product_name = row['product_name']
            product_description = row['product_description']
            
            dimensions, perishable, explosive = process_product(product_name, product_description)
            
            results.append({
                'Product Name': product_name,
                'Dimensions': dimensions,
                'Perishable': perishable,
                'Explosive': explosive
            })
        
        return jsonify(results), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
