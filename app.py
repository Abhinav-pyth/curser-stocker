import os
import json
import requests
import yfinance as yf
import plotly
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, request, flash
from flask_compress import Compress
from flask_caching import Cache
from flask_talisman import Talisman
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from config import config

load_dotenv()

# Initialize Flask app with configuration
app = Flask(__name__)
env = os.environ.get('FLASK_ENV', 'development')
app.config.from_object(config[env])

# Initialize extensions
Compress(app)
cache = Cache(app)
Talisman(app, content_security_policy=None)

# Initialize Hugging Face Inference Client
try:
    client = InferenceClient(
        provider="together",
        api_key=os.environ["HUGGINGFACE_API_KEY"]
    )
    print("Hugging Face Inference Client initialized successfully")
except Exception as e:
    print(f"Error initializing Inference Client: {str(e)}")
    client = None

def format_currency(value):
    if isinstance(value, (int, float)):
        return f"₹{value:,.2f}"
    return value

@cache.memoize(timeout=300)
def analyze_with_deepseek(stock_data, current_price, symbol):
    if client is None:
        return "DeepSeek model not available. Using alternative analysis."
    
    # Prepare financial metrics for analysis
    metrics = {
        'current_price': current_price,
        '200dma': stock_data['200DMA'].iloc[-1] if '200DMA' in stock_data.columns else None,
        'volume': stock_data['Volume'].iloc[-1],
        'high_52w': stock_data['High'].max(),
        'low_52w': stock_data['Low'].min(),
        'price_change': ((stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[0]) / stock_data['Close'].iloc[0]) * 100,
        'avg_volume': stock_data['Volume'].mean(),
        'volatility': stock_data['Close'].pct_change().std() * 100
    }
    
    prompt = f"""As an expert Indian stock market analyst, analyze {symbol} with these metrics:

Current Price: ₹{metrics['current_price']:.2f}
200 Day MA: ₹{metrics['200dma']:.2f}
52-Week High: ₹{metrics['high_52w']:.2f}
52-Week Low: ₹{metrics['low_52w']:.2f}
Price Change: {metrics['price_change']:.2f}%
Average Daily Volume: {metrics['avg_volume']:,.0f}
Volatility: {metrics['volatility']:.2f}%

Provide a comprehensive analysis with the following structure:

1. PRICE TARGETS (all in INR):
• Short-term (1-3 months): [specify target]
• Medium-term (3-6 months): [specify target]
• Long-term (6-12 months): [specify target]
• Stop-loss: [specify level]

2. TECHNICAL ANALYSIS:
• Current Trend: [Bullish/Bearish/Sideways]
• Support Levels: [List 2-3 levels]
• Resistance Levels: [List 2-3 levels]
• Moving Average Analysis
• Volume Analysis

3. RISK ASSESSMENT:
• Risk Level: [Low/Medium/High]
• Key Risk Factors
• Volatility Analysis
• Market Sentiment

4. TRADING RECOMMENDATION:
• Entry Points
• Exit Targets
• Position Sizing
• Time Horizon

Format the response with clear bullet points and specific price values in INR. Be precise with numbers and recommendations."""

    try:
        messages = [
            {
                "role": "system",
                "content": "You are an expert Indian stock market analyst specializing in technical analysis and price predictions."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        completion = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",
            messages=messages,
            max_tokens=800,
            temperature=0.7
        )
        
        # Extract the analysis from the completion
        analysis = completion.choices[0].message.content
        return analysis
    except Exception as e:
        return f"Error in DeepSeek analysis: {str(e)}"

@cache.memoize(timeout=300)
def get_stock_data(symbol):
    try:
        # Add .NS suffix for NSE stocks if not present
        if not symbol.endswith('.NS'):
            symbol = f"{symbol}.NS"
        
        # Get stock information
        stock = yf.Ticker(symbol)
        
        # Get historical data for the past year
        hist = stock.history(period="1y")
        
        if hist.empty:
            return None, "No data found for this stock symbol."
        
        # Calculate 200-day moving average
        hist['200DMA'] = hist['Close'].rolling(window=200).mean()
        
        # Get DeepSeek analysis
        current_price = hist['Close'].iloc[-1]
        deepseek_analysis = analyze_with_deepseek(hist, current_price, symbol[:-3])
        
        # Create candlestick chart with 200 DMA
        candlestick = go.Figure()
        
        # Add candlestick
        candlestick.add_trace(go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name='OHLC'
        ))
        
        # Add 200 DMA line
        candlestick.add_trace(go.Scatter(
            x=hist.index,
            y=hist['200DMA'],
            line=dict(color='orange', width=2),
            name='200 DMA'
        ))
        
        candlestick.update_layout(
            title=f'{symbol[:-3]} Stock Price - Past Year',
            yaxis_title='Stock Price (₹)',
            template='plotly_white',
            xaxis_rangeslider_visible=False,
            height=600,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Create volume chart with color based on price change
        colors = ['red' if row['Open'] - row['Close'] > 0 
                 else 'green' for index, row in hist.iterrows()]
        
        volume = go.Figure([go.Bar(
            x=hist.index,
            y=hist['Volume'],
            marker_color=colors,
            name='Volume'
        )])
        
        volume.update_layout(
            title=f'{symbol[:-3]} Trading Volume - Past Year',
            yaxis_title='Volume',
            template='plotly_white',
            height=300
        )
        
        # Get company info and financials
        info = stock.info
        
        # Convert market cap to INR (assuming USD to INR conversion rate of 83)
        usd_to_inr = 83
        market_cap_inr = info.get('marketCap', 0) * usd_to_inr if info.get('marketCap') else 'N/A'
        
        financials = {
            'Market Cap': format_currency(market_cap_inr) if market_cap_inr != 'N/A' else 'N/A',
            'PE Ratio': info.get('trailingPE', 'N/A'),
            'EPS': format_currency(info.get('trailingEps', 'N/A')),
            '52 Week High': format_currency(info.get('fiftyTwoWeekHigh', 'N/A')),
            '52 Week Low': format_currency(info.get('fiftyTwoWeekLow', 'N/A')),
            'Average Volume': f"{info.get('averageVolume', 0):,}",
            'Dividend Yield': f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get('dividendYield') else 'N/A',
        }
        
        # Get recent news
        news = []
        try:
            for item in stock.news[:5]:  # Get last 5 news items
                news.append({
                    'title': item.get('title', ''),
                    'publisher': item.get('publisher', ''),
                    'link': item.get('link', ''),
                    'published': datetime.fromtimestamp(item.get('providerPublishTime', 0)).strftime('%Y-%m-%d %H:%M')
                })
        except:
            news = []
        
        return {
            'candlestick': json.dumps(candlestick, cls=plotly.utils.PlotlyJSONEncoder),
            'volume': json.dumps(volume, cls=plotly.utils.PlotlyJSONEncoder),
            'company_name': info.get('longName', symbol[:-3]),
            'current_price': format_currency(info.get('currentPrice', 'N/A')),
            'currency': 'INR',
            'financials': financials,
            'description': info.get('longBusinessSummary', 'No description available.'),
            'news': news,
            'deepseek_analysis': deepseek_analysis
        }, None
    except Exception as e:
        return None, f"Error fetching data: {str(e)}"

def predict_stock(stock_symbol):
    # Get historical data for analysis
    stock = yf.Ticker(f"{stock_symbol}.NS")
    hist = stock.history(period="1y")
    current_price = hist['Close'].iloc[-1] if not hist.empty else 0
    
    # Build prompt for stock price prediction
    prompt = f"""Analyze {stock_symbol} (NSE: {stock_symbol}.NS) with current price ₹{current_price:.2f}. Provide a detailed analysis including:

1. Price Targets (in INR):
   - Short-term (1-3 months)
   - Medium-term (3-6 months)
   - Long-term (6-12 months)
   - Suggested stop-loss level

2. Technical Analysis:
   - Current trend
   - Key support and resistance levels
   - 200 DMA analysis
   - Volume analysis

3. Fundamental Analysis:
   - Company strengths and weaknesses
   - Sector outlook
   - Key growth drivers
   - Risk factors

4. Market Sentiment:
   - Recent developments
   - Industry trends
   - Economic factors affecting the stock

Format the price targets and levels clearly with bullet points. Provide specific price values in INR."""
    
    # Get Azure OpenAI configuration from environment variables
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment_id = os.getenv("AZURE_DEPLOYMENT_ID", "gpt-35-turbo")
    
    if not api_key or not endpoint:
        return "API key or endpoint not set in environment variables."
    
    headers = {"Content-Type": "application/json", "api-key": api_key}
    url = f"{endpoint}/openai/deployments/{deployment_id}/chat/completions?api-version=2023-05-15"
    
    data = {
        "messages": [
            {
                "role": "system", 
                "content": """You are an expert Indian stock market analyst. 
                Provide detailed analysis with specific price targets and clear recommendations. 
                Always include stop-loss levels and use bullet points for better readability.
                Focus on actionable insights and clear price levels."""
            },
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 800,
        "temperature": 0.7,
        "top_p": 0.95,
        "n": 1
    }
    
    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        return f"Error: {response.text}"
    
    result = response.json()
    return result['choices'][0]['message']['content'].strip()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        stock = request.form.get("stock")
        if not stock:
            flash("Please input a valid stock symbol.")
            return render_template("index.html")
            
        stock_data, error = get_stock_data(stock)
        if error:
            flash(error)
            return render_template("index.html")
            
        return render_template(
            "result.html",
            stock=stock,
            prediction=stock_data['deepseek_analysis'],
            stock_data=stock_data
        )
    
    return render_template("index.html")

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == "__main__":
    app.run(debug=True) 