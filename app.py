from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import json
import boto3
import pandas as pd
import requests
from io import BytesIO
from dotenv import load_dotenv
import os
from botocore.config import Config

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def hello_world():
    return 'Hello, World!'

# Set the configuration for the boto3 client
config = Config(
    retries={
        'max_attempts': 10,
        'mode': 'standard'
    },
    connect_timeout=60,
    read_timeout=60
)

# Initialize Bedrock client with the specified region and credentials
bedrock = boto3.client(
    'bedrock-runtime',
    region_name=os.getenv('AWS_REGION'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    aws_session_token=os.getenv('AWS_SESSION_TOKEN'),  # Add session token
    config=config  # Add the custom configuration
)

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
        
    except (json.JSONDecodeError, KeyError) as e:
        print(f"JSON or KeyError: {e}")
        raise e
    except Exception as e:
        print(f"Exception in process_product: {e}")
        raise e
    
    return dimensions, perishable, explosive

@app.route('/process', methods=['POST'])
def process_file():
    try:
        data = request.json
        file_url = data['fileUrl']

        response = requests.get(file_url)
        file_data = response.content

        df = pd.read_excel(BytesIO(file_data), engine='openpyxl')  # Specify the engine
        
        # Log the columns and the first few rows of the DataFrame for debugging
        print(f"DataFrame Columns: {df.columns}")
        print(f"DataFrame Head: {df.head()}")

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
        print(f"Error in process_file: {e}")  # Log the error
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
