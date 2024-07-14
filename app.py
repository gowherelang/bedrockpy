from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import boto3
import base64
import io
from PIL import Image
from botocore.config import Config
from botocore.exceptions import ClientError
import pandas as pd
import requests
from dotenv import load_dotenv
import os
from io import BytesIO

class ImageError(Exception):
    "Custom exception for errors returned by Amazon Titan Image Generator G1"

    def __init__(self, message):
        self.message = message

# Load environment variables from .env file
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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
    region_name='us-west-2',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    aws_session_token=os.getenv('AWS_SESSION_TOKEN'),  # Add session token
    config=config  # Add the custom configuration
)


@app.route('/')
def hello_world():
    return 'Hello, World!'

def process_product(product_name, product_description):
    print("Processing product:", product_name)
    prompt_data = (
        f"Provide product details for '{product_name}' in the following JSON format:\n"
        "{\n"
        f"    \"Product Name\": \"{product_name}\",\n"
        f"    \"Product Description\": \"{product_description}\",\n"
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
    
    print("Invoking model...")
    try:
        response = bedrock.invoke_model(
            body=body,
            modelId=model_id,
            accept="application/json",
            contentType="application/json",
        )
        print("Model invoked successfully!")
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
                'Product Description': product_description,
                'Dimensions': dimensions,
                'Perishable': perishable,
                'Explosive': explosive
            })
        
        return jsonify(results), 200
    except Exception as e:
        print(f"Error in process_file: {e}")  # Log the error
        return jsonify({'error': str(e)}), 500
    
    

def generate_image(prompt):
    model_id = 'amazon.titan-image-generator-v1'

    body = json.dumps({
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": prompt
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "height": 1024,
            "width": 1024,
            "cfgScale": 8.0,
            "seed": 0
        }
    })

    try:
        response = bedrock.invoke_model(
            body=body, modelId=model_id, accept="application/json", contentType="application/json"
        )
        response_body = json.loads(response.get("body").read())
        base64_image = response_body.get("images")[0]
        base64_bytes = base64_image.encode('ascii')
        image_bytes = base64.b64decode(base64_bytes)

        finish_reason = response_body.get("error")
        if finish_reason is not None:
            raise ImageError(f"Image generation error. Error is {finish_reason}")

        return image_bytes

    except ClientError as err:
        message = err.response["Error"]["Message"]
        print(f"A client error occurred: {message}")
        raise
    except ImageError as err:
        print(err.message)
        raise

@app.route('/generate', methods=['POST'])
def generate_image_endpoint():
    data = request.get_json()
    products = data['products']
    print("Products from POST request:", products)

    prompt = "Generate an image of an air cargo setup with the following parcels:\n\n"
    for product in products:
        prompt += f"Product: {product['Product Name']}\n"
        prompt += f"Description: {product['Product Description']}\n"
        prompt += f"Dimensions: {product['Dimensions']}\n"
        prompt += f"Perishable: {product['Perishable']}\n"
        prompt += f"Explosive: {product['Explosive']}\n"
        prompt += "\n"
    prompt += "The image should show the parcels arranged in the air cargo with their dimensions accurately represented."

    print("Prompt:", prompt)
    try:
        image_bytes = generate_image(prompt)
        image = Image.open(io.BytesIO(image_bytes))
        image_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode('utf-8')
        return jsonify({"image_url": image_url})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
