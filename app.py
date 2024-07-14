from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import boto3
import pandas as pd
import requests
from io import BytesIO
from dotenv import load_dotenv
import os
from botocore.config import Config
import base64
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

class ImageError(Exception):
    "Custom exception for errors returned by Amazon Titan Image Generator G1"

    def __init__(self, message):
        self.message = message

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
    region_name='us-west-2',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    aws_session_token=os.getenv('AWS_SESSION_TOKEN'),  # Add session token
    config=config  # Add the custom configuration
)

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
    
# Function to create a 3D box
def create_box(ax, origin, size, color='b'):
    x, y, z = origin
    dx, dy, dz = size
    xx = [x, x, x+dx, x+dx, x]
    yy = [y, y+dy, y+dy, y, y]
    kwargs = {'alpha': 0.8, 'color': color}
    ax.plot3D(xx, yy, [z]*5, **kwargs)
    ax.plot3D(xx, yy, [z+dz]*5, **kwargs)
    ax.plot3D([x, x], [y, y], [z, z+dz], **kwargs)
    ax.plot3D([x+dx, x+dx], [y, y], [z, z+dz], **kwargs)
    ax.plot3D([x, x], [y+dy, y+dy], [z, z+dz], **kwargs)
    ax.plot3D([x+dx, x+dx], [y+dy, y+dy], [z, z+dz], **kwargs)

def generate_matplotlib_image(products):
    # Setup 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Plot each product as a box
    origin = [0, 0, 0]
    for product in products:
        dimensions_str = product["Dimensions"].replace(" inches", "")
        dimensions = list(map(int, dimensions_str.split(' x ')))
        create_box(ax, origin, dimensions, color='b')
        origin[0] += dimensions[0] + 1  # Move to the next position

    # Save plot to BytesIO
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    return img_str

@app.route('/generate', methods=['POST'])
def generate_image():
    data = request.get_json()
    products = data['products']
    print("Products from POST request:", products)

    # Create a concise prompt for the GenAI model
    prompt = "Generate an image of an air cargo setup with the following parcels:\n\n"
    for product in products:
        prompt += f"Product: {product['Product Name']}\n"
        prompt += f"Dimensions: {product['Dimensions']}\n"
        prompt += "\n"
    prompt += (
        "Show the parcels arranged in the cargo area of an airplane as distinct, labeled boxes with accurate dimensions, similar to a 3D model used for cargo planning."
    )

    if len(prompt) > 512:
        return jsonify({"error": "Generated prompt is too long. Please reduce the number of products or simplify descriptions."}), 400

    print("Prompt:", prompt)
    try:
        print("Invoking model... generating image")
        # Call AWS Bedrock image generator model
        response = bedrock.invoke_model(
            modelId='amazon.titan-image-generator-v1',
            body=json.dumps({
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
            }),
            accept='application/json',
            contentType='application/json'
        )
        print("Model invoked successfully, image returning!")
        response_text = response['body'].read().decode('utf-8')
        response_json = json.loads(response_text)
        model_image_url = response_json['images'][0]  # Adjust this based on actual response

        # Generate Matplotlib image
        matplotlib_image = generate_matplotlib_image(products)

        return jsonify({"model_image_url": model_image_url, "matplotlib_image": matplotlib_image})

    except (boto3.exceptions.Boto3Error, Exception) as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
