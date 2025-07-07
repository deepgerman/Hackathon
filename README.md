# Hackathon
AI chatbot for an Electronics Marketplace that can provide all sorts of answers to consumers,
Creating a truly "full-fledged" AI chatbot involves several complex components: Natural Language Understanding (NLU) to interpret user intent, a robust backend to query databases (like Snowflake), dialogue management, and Natural Language Generation (NLG) to craft human-like responses.
-- Create a database and schema if they don't exist
CREATE DATABASE IF NOT EXISTS ELECTRONICS_SALES;
CREATE SCHEMA IF NOT EXISTS ELECTRONICS_SALES.PUBLIC;

-- Create the PRODUCTS table
CREATE OR REPLACE TABLE ELECTRONICS_SALES.PUBLIC.PRODUCTS (
    PRODUCT_ID INT,
    PRODUCT_NAME VARCHAR,
    CATEGORY VARCHAR, -- e.g., 'Laptop', 'Smartphone', 'Headphones', 'TV', 'Camera'
    BRAND VARCHAR,    -- e.g., 'Dell', 'Apple', 'Sony', 'Samsung', 'HP', 'Canon', 'LG'
    PRICE DECIMAL(10, 2),
    STOCK_QUANTITY INT,
    AVERAGE_RATING DECIMAL(2, 1), -- e.g., 4.5
    DESCRIPTION VARCHAR
);

-- Insert sample data
INSERT INTO ELECTRONICS_SALES.PUBLIC.PRODUCTS VALUES
(1, 'Dell XPS 13', 'Laptop', 'Dell', 1200.00, 50, 4.7, 'Powerful ultrabook with great display.'),
(2, 'MacBook Air M2', 'Laptop', 'Apple', 1499.00, 30, 4.8, 'Thin and light laptop with M2 chip.'),
(3, 'Sony WH-1000XM5', 'Headphones', 'Sony', 349.99, 120, 4.6, 'Industry-leading noise cancelling headphones.'),
(4, 'AirPods Pro 2', 'Headphones', 'Apple', 249.00, 200, 4.5, 'Active Noise Cancellation with Adaptive Transparency.'),
(5, 'Samsung Galaxy S24', 'Smartphone', 'Samsung', 799.00, 80, 4.4, 'Latest flagship Android phone from Samsung.'),
(6, 'iPhone 15 Pro', 'Smartphone', 'Apple', 999.00, 70, 4.7, 'Apple latest pro iPhone with A17 Bionic chip.'),
(7, 'HP Spectre x360', 'Laptop', 'HP', 1350.00, 40, 4.6, '2-in-1 convertible laptop.'),
(8, 'LG OLED C3', 'TV', 'LG', 1800.00, 25, 4.9, 'Stunning OLED TV with incredible picture quality.'),
(9, 'Canon EOS R5', 'Camera', 'Canon', 3899.00, 10, 4.8, 'High-resolution full-frame mirrorless camera.');


import os
import json
import snowflake.connector
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app) # Enable CORS for frontend communication

# --- Configuration ---
# Snowflake connection details from environment variables
SNOWFLAKE_ACCOUNT = os.getenv('SNOWFLAKE_ACCOUNT')
SNOWFLAKE_USER = os.getenv('SNOWFLAKE_USER')
SNOWFLAKE_PASSWORD = os.getenv('SNOWFLAKE_PASSWORD')
SNOWFLAKE_WAREHOUSE = os.getenv('SNOWFLAKE_WAREHOUSE')
SNOWFLAKE_DATABASE = os.getenv('SNOWFLAKE_DATABASE')
SNOWFLAKE_SCHEMA = os.getenv('SNOWFLAKE_SCHEMA')

# Gemini API Key from environment variable
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=GEMINI_API_KEY)

# Initialize the Gemini model
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# --- Snowflake Connection Pool (Basic Example - for production, consider a more robust pool) ---
def get_snowflake_connection():
    """Establishes and returns a Snowflake connection."""
    try:
        conn = snowflake.connector.connect(
            user=SNOWFLAKE_USER,
            password=SNOWFLAKE_PASSWORD,
            account=SNOWFLAKE_ACCOUNT,
            warehouse=SNOWFLAKE_WAREHOUSE,
            database=SNOWFLAKE_DATABASE,
            schema=SNOWFLAKE_SCHEMA
        )
        return conn
    except Exception as e:
        app.logger.error(f"Error connecting to Snowflake: {e}")
        return None

# --- Snowflake Data Retrieval Functions ---
def fetch_products(category=None, brand=None, price_max=None, rating_min=None, limit=5):
    """Fetches products from Snowflake based on criteria."""
    conn = get_snowflake_connection()
    if not conn:
        return []
    try:
        cursor = conn.cursor()
        sql_query = f"""
            SELECT PRODUCT_NAME, BRAND, PRICE, AVERAGE_RATING, STOCK_QUANTITY
            FROM {SNOWFLAKE_DATABASE}.{SNOWFLAKE_SCHEMA}.PRODUCTS
            WHERE 1=1
        """
        params = []

        if category:
            sql_query += " AND LOWER(CATEGORY) = LOWER(%s)"
            params.append(category)
        if brand:
            sql_query += " AND LOWER(BRAND) = LOWER(%s)"
            params.append(brand)
        if price_max is not None:
            sql_query += " AND PRICE <= %s"
            params.append(float(price_max))
        if rating_min is not None:
            sql_query += " AND AVERAGE_RATING >= %s"
            params.append(float(rating_min))

        sql_query += f" ORDER BY AVERAGE_RATING DESC, PRICE ASC LIMIT {limit};"
        
        cursor.execute(sql_query, params)
        results = cursor.fetchall()
        
        # Convert results to a list of dictionaries for easier processing
        columns = [desc[0] for desc in cursor.description]
        product_data = [dict(zip(columns, row)) for row in results]
        return product_data
    except Exception as e:
        app.logger.error(f"Error fetching products from Snowflake: {e}")
        return []
    finally:
        if conn:
            conn.close()

def get_product_details(product_name):
    """Fetches detailed information for a specific product."""
    conn = get_snowflake_connection()
    if not conn:
        return None
    try:
        cursor = conn.cursor()
        sql_query = f"""
            SELECT PRODUCT_NAME, CATEGORY, BRAND, PRICE, STOCK_QUANTITY, AVERAGE_RATING, DESCRIPTION
            FROM {SNOWFLAKE_DATABASE}.{SNOWFLAKE_SCHEMA}.PRODUCTS
            WHERE LOWER(PRODUCT_NAME) LIKE %s
            LIMIT 1;
        """
        cursor.execute(sql_query, [f"%{product_name.lower()}%"])
        result = cursor.fetchone()
        if result:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, result))
        return None
    except Exception as e:
        app.logger.error(f"Error fetching product details from Snowflake: {e}")
        return None
    finally:
        if conn:
            conn.close()

# --- Gemini AI Interaction ---
def get_gemini_response(user_message, context_data=None):
    """Sends user message and context to Gemini and gets a conversational response."""
    prompt_parts = [
        "You are an AI chatbot for an electronics marketplace. Your goal is to provide helpful, concise, and friendly answers to consumer questions about electronics products (e.g., Laptops, Smartphones, Headphones, TVs, Cameras, etc., from brands like Dell, Apple, Sony, Samsung, HP, Canon, LG).",
        "You have access to product data. When providing product information, always refer to the data provided to you.",
        "If you cannot find specific data, state that you couldn't find it.",
        "Here's the user's query: ",
        user_message,
        "\n\n"
    ]

    if context_data:
        prompt_parts.append("Here is relevant product data from our database:\n")
        prompt_parts.append(json.dumps(context_data, indent=2)) # Add structured data
        prompt_parts.append("\n\n")
    
    prompt_parts.append("Based on the user's query and the provided data (if any), formulate a helpful and concise response.")

    try:
        # Use generate_content for conversational turn
        response = gemini_model.generate_content("".join(prompt_parts))
        return response.text
    except Exception as e:
        app.logger.error(f"Error generating Gemini response: {e}")
        return "I'm sorry, I'm having trouble processing your request at the moment. Please try again later."

# --- Flask API Endpoint ---
@app.route('/chat', methods=['POST'])
def chat():
    """Handles incoming chat messages from the frontend."""
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"response": "No message provided."}), 400

    context_data = None
    bot_response = "I'm still learning and will get better at understanding your requests!"

    # --- Simulate Intent Recognition and Entity Extraction (Simplified) ---
    # In a real app, you'd use an NLU service (e.g., Dialogflow, Rasa) here
    # For this example, we use simple keyword matching to determine intent and extract entities.
    lower_message = user_message.lower()

    if "laptop" in lower_message or "computer" in lower_message or "notebook" in lower_message:
        category = "Laptop"
        if "dell" in lower_message: brand = "Dell"
        elif "apple" in lower_message: brand = "Apple"
        elif "hp" in lower_message: brand = "HP"
        else: brand = None
        
        price_match = [s for s in lower_message.split() if s.replace('$', '').replace(',', '').isdigit()]
        price_max = price_match[0] if price_match else None

        rating_match = [s for s in lower_message.split() if 'rating' in lower_message and s.replace('.', '').isdigit()]
        rating_min = rating_match[0] if rating_match else None

        context_data = fetch_products(category=category, brand=brand, price_max=price_max, rating_min=rating_min)
        if not context_data:
            bot_response = "I couldn't find any laptops matching your criteria."
        else:
            bot_response = get_gemini_response(user_message, context_data)

    elif "smartphone" in lower_message or "phone" in lower_message:
        category = "Smartphone"
        if "samsung" in lower_message: brand = "Samsung"
        elif "iphone" in lower_message or "apple" in lower_message: brand = "Apple"
        else: brand = None

        context_data = fetch_products(category=category, brand=brand)
        if not context_data:
            bot_response = "I couldn't find any smartphones matching your criteria."
        else:
            bot_response = get_gemini_response(user_message, context_data)

    elif "headphone" in lower_message or "earbud" in lower_message:
        category = "Headphones"
        if "sony" in lower_message: brand = "Sony"
        elif "apple" in lower_message: brand = "Apple"
        else: brand = None
        context_data = fetch_products(category=category, brand=brand)
        if not context_data:
            bot_response = "I couldn't find any headphones matching your criteria."
        else:
            bot_response = get_gemini_response(user_message, context_data)

    elif "tv" in lower_message or "television" in lower_message:
        category = "TV"
        if "lg" in lower_message: brand = "LG"
        else: brand = None
        context_data = fetch_products(category=category, brand=brand)
        if not context_data:
            bot_response = "I couldn't find any TVs matching your criteria."
        else:
            bot_response = get_gemini_response(user_message, context_data)

    elif "camera" in lower_message:
        category = "Camera"
        if "canon" in lower_message: brand = "Canon"
        else: brand = None
        context_data = fetch_products(category=category, brand=brand)
        if not context_data:
            bot_response = "I couldn't find any cameras matching your criteria."
        else:
            bot_response = get_gemini_response(user_message, context_data)

    elif "stock" in lower_message or "available" in lower_message:
        # Attempt to extract product name from message
        product_keywords = ["dell xps 13", "macbook air m2", "sony wh-1000xm5", "airpods pro 2", "samsung galaxy s24", "iphone 15 pro", "hp spectre x360", "lg oled c3", "canon eos r5"]
        found_product = next((p for p in product_keywords if p in lower_message), None)
        
        if found_product:
            product_details = get_product_details(found_product)
            if product_details:
                stock = product_details.get('STOCK_QUANTITY', 0)
                if stock > 0:
                    bot_response = f"Yes, the {product_details['PRODUCT_NAME']} is currently in stock with {stock} units available."
                else:
                    bot_response = f"I'm sorry, the {product_details['PRODUCT_NAME']} is currently out of stock."
            else:
                bot_response = f"I couldn't find stock information for '{found_product}'. Please specify the product name."
        else:
            bot_response = "Which product's stock would you like to check?"

    elif "details" in lower_message or "about" in lower_message:
        product_keywords = ["dell xps 13", "macbook air m2", "sony wh-1000xm5", "airpods pro 2", "samsung galaxy s24", "iphone 15 pro", "hp spectre x360", "lg oled c3", "canon eos r5"]
        found_product = next((p for p in product_keywords if p in lower_message), None)

        if found_product:
            product_details = get_product_details(found_product)
            if product_details:
                bot_response = get_gemini_response(user_message, product_details)
            else:
                bot_response = f"I couldn't find details for '{found_product}'. Please specify the product name."
        else:
            bot_response = "Which product would you like details about?"

    else:
        # If no specific intent matched, send the raw message to Gemini for general conversation
        bot_response = get_gemini_response(user_message)

    return jsonify({"response": bot_response})
