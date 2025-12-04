from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
from pydantic import BaseModel
from typing import List, Dict, Optional
import pickle
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import uvicorn
from pydantic import BaseModel, EmailStr
from fastapi_mail import ConnectionConfig, FastMail, MessageSchema, MessageType
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
from dotenv import load_dotenv
import os
import google.generativeai as genai
import openai
import json


load_dotenv() 

import psycopg2
import pandas as pd

conn = psycopg2.connect(
    host="aws-0-ap-southeast-1.pooler.supabase.com",
    #port=6543,
    port=5432,
    database="postgres",
    user="postgres.mnxcvpzefuwuccprqnxl",
    password=os.getenv("PASS_DB"),
    sslmode="require"
)

def query_to_dataframe(conn, query):
    try:
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        colnames = [desc[0] for desc in cur.description]
        df = pd.DataFrame(rows, columns=colnames)
        cur.close()
        return df
    except Exception as e:
        print("Error executing query:", e)
        return pd.DataFrame() 


sender_email = "ahmadimanh3@gmail.com"  # fix the double @@
app_password = os.getenv("APP_PASSWORD")
smtp_server = "smtp.gmail.com"
port = 587

top_10_products_ts = pd.read_csv("csv_files\\top10_product_ts.csv")
top_10_products_forecast = pd.read_csv("csv_files\\top10_products_forecast_results.csv")
forecasted_orders = pd.read_csv("csv_files\\forecasted_orders_synthetic_1.csv")
orders_weekly = pd.read_csv("csv_files\\synthetic_orders_1.csv")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class EmailRequest(BaseModel):
    email: EmailStr
    item : str
    quantity: int

@app.post("/send-email-stock-alert")
async def send_stock_email(request: EmailRequest) -> bool:
    receiver_email = request.email
    subject = "Request to buy more stocks"

    # HTML email body
    body = f"""
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6;">
        <h2 style="color: #2E86C1;">Stock Refill Request</h2>
        <p>Dear Supplier,</p>
        <p>We would like to request a restock for the following item:</p>
        <table style="border-collapse: collapse; margin: 20px 0;">
        <tr>
            <td style="padding: 8px; font-weight: bold;">Item:</td>
            <td style="padding: 8px;">{request.item}</td>
        </tr>
        <tr>
            <td style="padding: 8px; font-weight: bold;">Quantity:</td>
            <td style="padding: 8px;">{request.quantity}</td>
        </tr>
        </table>
        <p>Please process this request as soon as possible. Thank you!</p>
        <p>Best regards,<br>AdventureWorks</p>
    </body>
    </html>
    """

    # Create the email message
    message = MIMEMultipart("alternative")
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "html"))

    try:
        with smtplib.SMTP(smtp_server, port) as server:
            server.starttls()
            print(app_password)
            server.login(sender_email, app_password)
            server.sendmail(sender_email, receiver_email, message.as_string())
        return True
    except Exception as e:
        print(f"Email send error: {e}")
        return False

@app.get("/main_page", response_class=HTMLResponse)
async def serve_step1_business_info(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})

@app.get("/details_page", response_class=HTMLResponse)
async def serve_details(request: Request):
    return templates.TemplateResponse("details.html", {"request": request})

@app.get("/orders_page", response_class=HTMLResponse)
async def serve_orders(request: Request):
    return templates.TemplateResponse("orders.html", {"request": request})


@app.get("/product-change-status")
async def product_change_status(product: str):
    return increase_or_decrease(product)



def increase_or_decrease(product):
    sales_data = top_10_products_ts
    forecast_data = top_10_products_forecast
    sales_data = sales_data[sales_data["ProductName"]==product]
    forecast_data = forecast_data[forecast_data["Product"]==product]
    sales_data = sales_data.iloc[:-1]
    last_4_weeks_sale = sum(sales_data.iloc[-4:]["sum"])
    forecast_lower_limit = sum(forecast_data.iloc[:4]["Lower 80%"])
    forecast_upper_limit = sum(forecast_data.iloc[:4]["Upper 80%"])
    prediction_status = "Decrease"
    if (last_4_weeks_sale <  forecast_lower_limit):
        prediction_status = "Increase"
    elif (last_4_weeks_sale < forecast_upper_limit and last_4_weeks_sale > forecast_lower_limit):
        prediction_status = "No significant changes"
    else :
        prediction_status = "Decrease"
    
    return_dict ={
        "last_4_weeks_sale":last_4_weeks_sale,
        "forecast_lower_limit":forecast_lower_limit,
        "forecast_upper_limit":forecast_upper_limit,
        "prediction_status":prediction_status
    }

    return return_dict

@app.get("/get_unique_products")
async def unique_products():
    sales_data = top_10_products_ts
    return sales_data["ProductName"].unique().tolist()

@app.get("/restock_or_not")
async def restock_or_not(product):
    query = """
    Select * from "stock_table"    
    """
    print("000000000000000000000000000000000000000000000")
    stock_table = query_to_dataframe(conn,query)
    stocks = stock_table[stock_table["Products"]==product].iloc[0]["Stock"]
    upper_limit = round(float(increase_or_decrease(product)["forecast_upper_limit"]), 2)
    lower_limit = round(float(increase_or_decrease(product)["forecast_lower_limit"]), 2)
    print("11111111111111111111111111111111111111111111111111")
    restock = False

    if(stocks < upper_limit):
        restock = True

    return {
        "Stocks": int(stocks),
        "Upper_Limit": float(upper_limit),
        "Predictions": f"{float(lower_limit)} - {float(upper_limit)}",
        "need_restock": bool(restock)
    }

@app.get("/linegraph_endpoint")
async def linegraph_endpoint(product: str):
    historical_data = top_10_products_ts[top_10_products_ts["ProductName"] == product]
    historical_data = historical_data.iloc[:-1]
    forecasted_data = top_10_products_forecast[top_10_products_forecast["Product"] == product]

    return {
        "historical_data": historical_data.to_dict(orient="records"),
        "forecasted_data": forecasted_data.to_dict(orient="records")
    }


@app.get("/linegraph_endpoint_orders")
async def linegraph_endpoint():
    historical_data = orders_weekly 
    historical_data = historical_data.iloc[:-1]
    forecasted_data = forecasted_orders

    return {
        "historical_data": historical_data.to_dict(orient="records"),
        "forecasted_data": forecasted_data.to_dict(orient="records")
    }

@app.get("/adjust_schedule")
async def adjust_schedule():
    staff_names = [
        "Alicia Tan", "Jason Koh", "Bryan Lim", "Chloe Wong", "Farah Aziz",
        "Daniel Lee", "Natalie Cheng", "Amir Hakim", "Stephanie Ong", "Kelvin Yap"
    ]

    df_schedule = pd.read_csv("csv_files\\Staff_schedule_nextweek.csv")

    prompt = f"""
    This is the staff schedule for next week
    {df_schedule}

    This is the staff list:
    {staff_names}

    It seems like the shop is getting busier next week, I need you to help me reschedule by adding more staff at each session.
    Your reply should only consist of Python code that creates the new schedule in a pandas DataFrame.
    The final dataframe should be stored in a variable called final_df.
    """

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)

    code_str = response.text.replace("```python", "").replace("```", "").strip()

    namespace = {"pd": pd, "staff_names": staff_names, "df_schedule": df_schedule}

    exec(code_str, namespace)

    final_df = namespace.get("final_df")

    # Convert DataFrame to JSON-friendly format
    return {"schedule": final_df.to_dict(orient="records")}



api_key = os.getenv("OPENAI_API")

# Create a client
client = openai.OpenAI(api_key=api_key)

def call_openai(prompt: str, model: str = "gpt-4o") -> str:
    """
    Send a prompt to OpenAI API (v1.0+ Python SDK) and return the generated text.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        # Extract the generated text
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return ""

def get_tables_sales_data():
    conn = psycopg2.connect(
    host="aws-0-ap-southeast-1.pooler.supabase.com",
    #port=6543,
    port=5432,
    database="postgres",
    user="postgres.mnxcvpzefuwuccprqnxl",
    password=os.getenv("PASS_DB"),
    sslmode="require"
    )
    query = '''
    Select * from "Sales_data"
    '''
    df_sales= query_to_dataframe(conn, query)
    return df_sales


def get_tables_products_data():
    conn = psycopg2.connect(
    host="aws-0-ap-southeast-1.pooler.supabase.com",
    #port=6543,
    port=5432,
    database="postgres",
    user="postgres.mnxcvpzefuwuccprqnxl",
    password=os.getenv("PASS_DB"),
    sslmode="require"
    )
    query = '''
    Select * from "Product_Lookup"
    '''
    df= query_to_dataframe(conn, query)
    return df


def get_tables_territory_data():
    conn = psycopg2.connect(
    host="aws-0-ap-southeast-1.pooler.supabase.com",
    #port=6543,
    port=5432,
    database="postgres",
    user="postgres.mnxcvpzefuwuccprqnxl",
    password=os.getenv("PASS_DB"),
    sslmode="require"
    )
    query = '''
    Select * from "Territory_Lookup"
    '''
    df= query_to_dataframe(conn, query)
    return df



def get_tables_subcategory_data():
    conn = psycopg2.connect(
    host="aws-0-ap-southeast-1.pooler.supabase.com",
    #port=6543,
    port=5432,
    database="postgres",
    user="postgres.mnxcvpzefuwuccprqnxl",
    password=os.getenv("PASS_DB"),
    sslmode="require"
    )
    query = '''
    Select * from "Product_Subcategories_Lookup"
    '''
    df= query_to_dataframe(conn, query)
    return df


FORBIDDEN_NAMES = { "sys", "subprocess", "shutil", "__import__"}


import ast
def is_code_safe(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        print("Syntax error in code:", e)
        return False

    for node in ast.walk(tree):
        # Check for dangerous imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names]
            if any(name.split('.')[0] in FORBIDDEN_NAMES for name in names):
                print(f"Found forbidden import in code: {names}")
                return False

        # Check for dangerous function calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec", "__import__"}:
                print(f"Found dangerous function call: {node.func.id}")
                return False

    return True

def gemini_reply(prompt):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text


@app.get("/chatbot_endpoint")
async def generate_reply(user_query):
    prompt = f"""
    
    You are an AI assistant that will help answer user's question about a dataset, you need to answer the user's question about the data, so here is the data

    Firstly assume that today is 30th June 2022
    the first table is sales data (its like transactions) and these are the columns:
    - OrderDate (String) : example 30/6/2020 is 30th June 2020
    - OrderNumber (String) this is a unique identifer of each orders, but this is not a primary key as there can be two instances rows with the same order Number for example if they purchase more than one product
    - ProductKey(int) this is the product id of the product bought (also used a column to merge product lookup and sales data)
    - OrderQuantity(int) the quantity of product ordered
    - TerritoryKey(int) a unique id of from where the product is bought

    to get the dataframe for sales data:
    sales_df = get_tables_sales_data(), i already defined this you dont need to define it yourself

    the second table is the product_lookup table and these are the columns:
    - ProductKey(int)  like primary key (connect sales data with product lookup table)
    - ProductName (str)
    - ModelName (str)
    - ProductColor
    - ProductCost (float) how much does it cost to make the product
    - ProductPrice (float) how much do we sell it
    - ProductSubcategoryKey(int) a key connecting this table to the product_subcategories lookup table

    these are the uniquevalues for the ProductName
    {get_tables_products_data()["ProductName"].unique()}

    and these are the unique values for the ModelName
    {get_tables_products_data()["ModelName"].unique()}

    to get the dataframe Product_lookup:
    product_lookup = get_tables_products_data(), i already defined this you dont need to define it yourself

    the third table is the territory_lookup table, the columns are
    - SalesTerritoryKey(int) primary key (connect sales data with territory lookup table)
    - Country(string) actual name of the place (actually here it is state not country cuz the values here are Kuala Lumpur , Selangor etc)

    to get the dataframe Territory_Lookup:
    territory_lookup = get_tables_territory_data(), i already defined this you dont need to define it yourself

    the fourth table is the product_subcategory_lookup, the columns are:
    - ProductCategoryKey(int) a key to connect this table with the Product Category lookup table
    - SubcategoryName
    - ProductSubcategoryKey(int) (this is like the unique identifier)

    These are the unique values for the subcategoryName
    ['Mountain Bikes', 'Road Bikes', 'Touring Bikes', 'Handlebars',
       'Bottom Brackets', 'Brakes', 'Chains', 'Cranksets', 'Derailleurs',
       'Forks', 'Headsets', 'Mountain Frames', 'Pedals', 'Road Frames',
       'Saddles', 'Touring Frames', 'Wheels', 'Bib-Shorts', 'Caps',
       'Gloves', 'Jerseys', 'Shorts', 'Socks', 'Tights', 'Vests',
       'Bike Racks', 'Bike Stands', 'Bottles and Cages', 'Cleaners',
       'Fenders', 'Helmets', 'Hydration Packs', 'Lights', 'Locks',
       'Panniers', 'Pumps', 'Tires and Tubes']

    to get the dataframe Subcategory_Lookup:
    subcategroy_lookup = get_tables_subcategory_data(), i already defined this you dont need to define it yourself

    

    DONT MAKE ASSUMPTIONS

    Now your task is to write a python code to answer the user's question.
    Your reply should only be python code and nothing else.
    Your final output must be stored in a variable named final_output.

    One more thing , dont ever make a graph where the x-axis or the y-axis is every unique product cuz it will make the graph so ugly

    if a graph is needed to answer the question ,draw it t and  the final_output should be a dictionary version of the graph

    in the python code no need to do plt.show()

  
    """

    # Get Python code from OpenAI
    code = call_openai(prompt + "\nUser query: " + user_query)

    # Remove code fences
    code = (
        code.replace("```python", "")
               .replace("```", "")
               .strip()
    )

    print("AI Generated Code:")
    print(code)

    final_output=" "

    # Prepare local namespace for AI execution
    exec_globals={}
    try:
        if is_code_safe(code):
            exec_globals["get_tables_sales_data"] = get_tables_sales_data
            exec_globals["get_tables_products_data"] = get_tables_products_data
            exec_globals["get_tables_territory_data"] = get_tables_territory_data
            exec_globals["get_tables_subcategory_data"] = get_tables_subcategory_data
            exec(code,exec_globals)
            final_output = exec_globals.get('final_output')
        else:
            print("There is something wrong")
    except Exception as e:
        print(f"Error: {e}")



    prompt2 = f"""
    User's Question : {user_query}
    Based on the question the AI tried to answer it using this code
    {code}

    and got this 
    final_output = {final_output}

    if the final_output is just a value than dont try to draw a graph
    
    if the final_output is a dictionary version of a graph, write me some hardcoded chart js code to replicate that graph (use what color you wanted)

    now based on the final_output, provide a necessary answer to the user's question, your output should be in this structure for example

    {{
    explanation : "The graph showsss..." or if there is no graph and just direct answers "the revenue for march 2020 is.." for example,
    chartjscode : "new Chart(document.getElementById('chartCanvas'), {{...}})...;, None if there is no graph involed"
    have_graph: True if there is a graph, False if there is no graph
    }}

    YOU SHOULD ONLY RETURN A JSON STYLE OUTPUT AND NOTHING ELSE


    """

    final_reply = gemini_reply(prompt2)

    final_reply = (
        final_reply.replace("```json", "")
               .replace("```", "")
               .strip()
    )

    try:
        final_json = json.loads(final_reply)
    except json.JSONDecodeError as e:
        print("JSON Parse Error:", e)
        print("Raw text was:", final_reply)
        return None

    return final_json


  



#########################################################################################################################################

#Code BOSS

# Database connection
def get_db_connection():
    """Create database connection"""
    return psycopg2.connect(
    host="aws-0-ap-southeast-1.pooler.supabase.com",
    #port=6543,
    port=5432,
    database="postgres",
    user="postgres.mnxcvpzefuwuccprqnxl",
    password=os.getenv("PASS_DB"),
    sslmode="require"
)

# ============ ROUTES ============

@app.get("/", response_class=HTMLResponse)
async def analytics_home(request: Request):
    """Serve the analytics dashboard"""
    return templates.TemplateResponse("analytics.html", {"request": request})


# ============ API ENDPOINTS ============

@app.get("/api/analytics/overview")
async def get_analytics_overview():
    """Get overview KPIs with period comparison"""
    try:
        conn = get_db_connection()
        
        # Get max date from data
        date_query = """
        SELECT MAX(TO_DATE("OrderDate", 'DD/MM/YYYY')) as max_date
        FROM "Sales_data"
        """
        df_date = pd.read_sql(date_query, conn)
        max_date = df_date['max_date'].iloc[0]
        
        # Current period (last 30 days of available data)
        query_current = f"""
        SELECT 
            COUNT(DISTINCT "OrderNumber") as total_orders,
            COALESCE(SUM("OrderQuantity"), 0) as total_units,
            COUNT(DISTINCT "CustomerKey") as unique_customers
        FROM "Sales_data"
        WHERE TO_DATE("OrderDate", 'DD/MM/YYYY') >= '{max_date}'::date - INTERVAL '30 days'
        """
        
        df_current = pd.read_sql(query_current, conn)
        
        # Previous period (30-60 days before max date)
        query_previous = f"""
        SELECT 
            COUNT(DISTINCT "OrderNumber") as total_orders,
            COALESCE(SUM("OrderQuantity"), 0) as total_units,
            COUNT(DISTINCT "CustomerKey") as unique_customers
        FROM "Sales_data"
        WHERE TO_DATE("OrderDate", 'DD/MM/YYYY') >= '{max_date}'::date - INTERVAL '60 days'
        AND TO_DATE("OrderDate", 'DD/MM/YYYY') < '{max_date}'::date - INTERVAL '30 days'
        """
        
        df_previous = pd.read_sql(query_previous, conn)
        conn.close()
        
        current = {
            "total_orders": int(df_current['total_orders'].iloc[0] or 0),
            "total_units": int(df_current['total_units'].iloc[0] or 0),
            "unique_customers": int(df_current['unique_customers'].iloc[0] or 0)
        }
        
        previous = {
            "total_orders": int(df_previous['total_orders'].iloc[0] or 0),
            "total_units": int(df_previous['total_units'].iloc[0] or 0),
            "unique_customers": int(df_previous['unique_customers'].iloc[0] or 0)
        }
        
        def safe_change(curr, prev):
            if prev == 0:
                return 100.0 if curr > 0 else 0.0
            return round(((curr - prev) / prev) * 100, 2)
        
        changes = {
            "orders": safe_change(current["total_orders"], previous["total_orders"]),
            "units": safe_change(current["total_units"], previous["total_units"]),
            "customers": safe_change(current["unique_customers"], previous["unique_customers"])
        }
        
        return {
            "current_period": current,
            "previous_period": previous,
            "changes": changes
        }
        
    except Exception as e:
        print(f"Error in overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/sales-trend")
async def get_sales_trend(weeks: int = 12):
    """Get weekly sales trend"""
    try:
        conn = get_db_connection()
        
        # Get max date
        date_query = """SELECT MAX(TO_DATE("OrderDate", 'DD/MM/YYYY')) as max_date FROM "Sales_data" """
        df_date = pd.read_sql(date_query, conn)
        max_date = df_date['max_date'].iloc[0]
        
        query = f"""
        SELECT 
            DATE_TRUNC('week', TO_DATE("OrderDate", 'DD/MM/YYYY')) as week_start,
            SUM("OrderQuantity") as total_units,
            COUNT(DISTINCT "OrderNumber") as order_count
        FROM "Sales_data"
        WHERE TO_DATE("OrderDate", 'DD/MM/YYYY') >= '{max_date}'::date - INTERVAL '{weeks} weeks'
        GROUP BY week_start
        ORDER BY week_start
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        trend_data = []
        for _, row in df.iterrows():
            trend_data.append({
                "week": row['week_start'].strftime('%Y-%m-%d'),
                "units": int(row['total_units']),
                "orders": int(row['order_count'])
            })
        
        return {
            "data": trend_data,
            "total_weeks": len(trend_data)
        }
        
    except Exception as e:
        print(f"Error in sales trend: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/top-products")
async def get_top_products(limit: int = 10):
    """Get top selling products"""
    try:
        conn = get_db_connection()
        
        # Get max date
        date_query = """SELECT MAX(TO_DATE("OrderDate", 'DD/MM/YYYY')) as max_date FROM "Sales_data" """
        df_date = pd.read_sql(date_query, conn)
        max_date = df_date['max_date'].iloc[0]
        
        query = f"""
        SELECT 
            p."ProductName",
            COALESCE(pc."CategoryName", 'Uncategorized') as category_name,
            SUM(s."OrderQuantity") as total_sold,
            COUNT(DISTINCT s."OrderNumber") as order_frequency
        FROM "Sales_data" s
        JOIN "Product_Lookup" p ON s."ProductKey" = p."ProductKey"
        LEFT JOIN "Product_Subcategories_Lookup" psc ON p."ProductSubcategoryKey" = psc."ProductSubcategoryKey"
        LEFT JOIN "Product_Category_Lookup" pc ON psc."ProductCategoryKey" = pc."ProductCategoryKey"
        WHERE TO_DATE(s."OrderDate", 'DD/MM/YYYY') >= '{max_date}'::date - INTERVAL '30 days'
        GROUP BY p."ProductName", pc."CategoryName"
        ORDER BY total_sold DESC
        LIMIT {limit}
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        if df.empty:
            return {"products": [], "total": 0}
        
        products = []
        for _, row in df.iterrows():
            products.append({
                "product": str(row['ProductName']) if pd.notna(row['ProductName']) else 'Unknown',
                "category": str(row['category_name']) if pd.notna(row['category_name']) else 'Uncategorized',
                "units_sold": int(row['total_sold']) if pd.notna(row['total_sold']) else 0,
                "orders": int(row['order_frequency']) if pd.notna(row['order_frequency']) else 0
            })
        
        return {
            "products": products,
            "total": len(products)
        }
        
    except Exception as e:
        print(f"Error in top products: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/category-breakdown")
async def get_category_breakdown():
    """Get sales distribution by category"""
    try:
        conn = get_db_connection()
        
        # Get max date
        date_query = """SELECT MAX(TO_DATE("OrderDate", 'DD/MM/YYYY')) as max_date FROM "Sales_data" """
        df_date = pd.read_sql(date_query, conn)
        max_date = df_date['max_date'].iloc[0]
        
        query = f"""
        SELECT 
            COALESCE(pc."CategoryName", 'Uncategorized') as category_name,
            SUM(s."OrderQuantity") as total_units,
            COUNT(DISTINCT s."OrderNumber") as order_count,
            COUNT(DISTINCT s."ProductKey") as product_count
        FROM "Sales_data" s
        JOIN "Product_Lookup" p ON s."ProductKey" = p."ProductKey"
        LEFT JOIN "Product_Subcategories_Lookup" psc ON p."ProductSubcategoryKey" = psc."ProductSubcategoryKey"
        LEFT JOIN "Product_Category_Lookup" pc ON psc."ProductCategoryKey" = pc."ProductCategoryKey"
        WHERE TO_DATE(s."OrderDate", 'DD/MM/YYYY') >= '{max_date}'::date - INTERVAL '30 days'
        GROUP BY pc."CategoryName"
        ORDER BY total_units DESC
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        if df.empty:
            return {"categories": [], "total_units": 0}
        
        total_units = df['total_units'].sum()
        
        categories = []
        for _, row in df.iterrows():
            categories.append({
                "category": str(row['category_name']) if pd.notna(row['category_name']) else 'Uncategorized',
                "units": int(row['total_units']) if pd.notna(row['total_units']) else 0,
                "orders": int(row['order_count']) if pd.notna(row['order_count']) else 0,
                "products": int(row['product_count']) if pd.notna(row['product_count']) else 0,
                "percentage": round((row['total_units'] / total_units) * 100, 2) if total_units > 0 else 0
            })
        
        return {
            "categories": categories,
            "total_units": int(total_units) if pd.notna(total_units) else 0
        }
        
    except Exception as e:
        print(f"Error in category breakdown: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/territory-performance")
async def get_territory_performance(limit: int = 10):
    """Get sales by territory - aggregated by country"""
    try:
        conn = get_db_connection()
        
        # Get max date
        date_query = """SELECT MAX(TO_DATE("OrderDate", 'DD/MM/YYYY')) as max_date FROM "Sales_data" """
        df_date = pd.read_sql(date_query, conn)
        max_date = df_date['max_date'].iloc[0]
        
        # UPDATED QUERY: Aggregate by Country only (combines all US regions)
        query = f"""
        SELECT 
            t."Country",
            SUM(s."OrderQuantity") as total_units,
            COUNT(DISTINCT s."OrderNumber") as order_count
        FROM "Sales_data" s
        JOIN "Territory_Lookup" t ON s."TerritoryKey" = t."SalesTerritoryKey"
        WHERE TO_DATE(s."OrderDate", 'DD/MM/YYYY') >= '{max_date}'::date - INTERVAL '30 days'
        GROUP BY t."Country"
        ORDER BY total_units DESC
        LIMIT {limit}
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        if df.empty:
            return {"territories": [], "total": 0, "total_units": 0}
        
        total_units = df['total_units'].sum()
        
        territories = []
        for _, row in df.iterrows():
            territories.append({
                "country": str(row['Country']) if pd.notna(row['Country']) else 'Unknown',
                "units": int(row['total_units']) if pd.notna(row['total_units']) else 0,
                "orders": int(row['order_count']) if pd.notna(row['order_count']) else 0,
                "percentage": round((row['total_units'] / total_units) * 100, 2) if total_units > 0 else 0
            })
        
        return {
            "territories": territories,
            "total": len(territories),
            "total_units": int(total_units) if pd.notna(total_units) else 0
        }
        
    except Exception as e:
        print(f"Error in territory performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    

