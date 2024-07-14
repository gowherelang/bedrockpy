# AWSupabase Generative AI Hackathon

## Overview

This project is part of the AWS Supabase Hackathon. It aims to provide a comprehensive solution for airlines to manage their cargo logistics efficiently by leveraging advanced AI technology provided by AWS services and Supabase. The app generates detailed product information including dimensions, perishable status, and explosive status, and visualizes cargo arrangements in an airplane.

## Problem Statement

Airlines face numerous challenges when managing cargo logistics, such as:
- Ensuring efficient use of cargo space without exceeding weight and size limits.
- Managing the logistics of perishable goods to ensure timely delivery.
- Handling potentially explosive items safely and adhering to safety regulations.

## Solution

Our solution addresses these challenges by providing detailed insights into product characteristics, helping airlines make informed decisions about cargo allocation. Here is a step-by-step breakdown:

1. **Input Collection**: Users enter the product name and description via a user-friendly interface.
2. **Data Processing**: The input data is sent to an AI model that generates detailed product information.
3. **Detailed Insights**: The AI model provides dimensions, perishable status, and explosive status of the items.
4. **Output**: The generated details are displayed to the user, and an image of the product in the cargo is also generated.

## Technologies Used

- **AWS**: Amazon Bedrock
  - Models used:
    - Text: amazon.titan-text-express-v1
    - Image: Amazon Titan Image Generator G1
- **Supabase**: Supabase Storage, Supabase Database

## Demo

Watch our demo [here](https://drive.google.com/file/d/1jn1NssFQ_AmSza-Go44gUDQjsPf9_Zpp/view?usp=sharing).

## Use Case

### Scenario

An airline receives a large shipment containing various items, including electronics, fresh produce, and chemicals. The logistics team uses the report to load the cargo efficiently, ensuring compliance with safety regulations and maximizing cargo space usage.

### Steps

1. **Input**: The shipment details are uploaded to the system.
2. **Processing**: The AI model analyzes each item and generates detailed information about dimensions, perishability, and explosiveness.
3. **Output**: The system provides a comprehensive report suggesting optimal cargo allocation:
   - Fresh produce is prioritized for the next available flight to ensure timely delivery.
   - Electronics are allocated based on available space and weight distribution.
   - Chemicals with explosive properties are flagged and assigned to a cargo-only flight with necessary safety measures.
4. **Action**: The logistics team uses the report to load the cargo efficiently.

## Scaling in the Real World

Our solution can be scaled and adapted for real-world applications by:
- Handling larger and more complex parcels through batch processing.
- Implementing real-time data processing to handle last-minute changes in cargo lists and flight schedules.

## How to Run the Backend

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
