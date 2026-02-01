
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import shap
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import src.data_loader as dl
import src.preprocessing as pp
import src.explainability as xai
import src.llm_utils as llm
import math

def sanitize_json(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    return obj

# Load Env
load_dotenv('.env.local')

app = FastAPI(title="Market Intelligence API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# GLOBAL STATE
# -----------------------------------------------------------------------------
MODEL = None
FEATURE_COLS = None
AVAILABLE_TICKERS = set()

def load_available_tickers():
    global AVAILABLE_TICKERS
    try:
        # Try to load from local cache first to save API calls
        # For now, let's try to fetch from DL or assume a dynamic list.
        # Actually, let's try to list files in data directory if they are cached? No.
        # Let's use the DB approach.
        tickers = dl.get_available_tickers()
        if tickers:
            AVAILABLE_TICKERS = set(t.upper() for t in tickers)
            print(f"Loaded {len(AVAILABLE_TICKERS)} available tickers.")
        else:
            # Fallback list for demo if DB fetch fails
            print("Warning: Could not load dynamic ticker list. Using fallback.")
            AVAILABLE_TICKERS = {"AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "JPM", "V", "JNJ", "WMT", "PG", "XOM"}
    except Exception as e:
        print(f"Error loading tickers: {e}")

def load_model():
    global MODEL, FEATURE_COLS
    if MODEL is None:
        try:
            print("Loading Model...")
            model = xgb.XGBClassifier()
            model.load_model("output/latest_model.json")
            
            # Use the model's internal feature names as the source of truth
            booster = model.get_booster()
            feature_cols = booster.feature_names
            
            MODEL = model
            FEATURE_COLS = feature_cols
            print(f"Model Loaded. Expecting {len(feature_cols)} features.")
        except Exception as e:
            print(f"Error loading model: {e}")

@app.on_event("startup")
async def startup_event():
    load_model()
    load_available_tickers()

# -----------------------------------------------------------------------------
# ENDPOINTS
# -----------------------------------------------------------------------------

@app.get("/")
def read_root():
    return {"status": "ok", "service": "Market Intelligence API"}

@app.get("/tickers")
def get_tickers():
    """Return all locally available tickers."""
    return sorted(list(AVAILABLE_TICKERS))

@app.get("/search")
def search_assets(q: str):
    """
    Search for assets via Yahoo Finance API (YahooQuery) and filter by local availability.
    """
    try:
        from yahooquery import search
        
        if not q:
            return []
            
        # 1. Search via Yahoo
        results = search(q) # Returns dict: {'quotes': [...], ...}
        
        if 'quotes' not in results:
            return []
            
        candidates = results['quotes']
        
        # 2. Filter matches
        # Check if the symbol exists in our AVAILABLE_TICKERS set
        # If we have a local set, filter. If local set is empty/failed, return all (or top) but mark as 'unknown availability'.
        
        filtered = []
        for cand in candidates:
            symbol = cand.get('symbol', '').upper()
            shortname = cand.get('shortname', '')
            longname = cand.get('longname', '')
            
            # Skip if no symbol
            if not symbol:
                continue
                
            # Check availability
            is_available = symbol in AVAILABLE_TICKERS
            
            # IF user strictly wants ONLY available:
            if is_available:
                 filtered.append({
                    "symbol": symbol,
                    "name": shortname or longname or symbol,
                    "exchange": cand.get('exchange', ''),
                    "type": cand.get('quoteType', '')
                })
        
        return filtered
        
    except Exception as e:
        print(f"Search error: {e}")
        return []


@app.get("/analysis/{ticker}")
def analyze_ticker(ticker: str):
    """
    Full analysis pipeline for a ticker:
    1. Fetch Data
    2. Preprocess
    3. Predict
    4. Explain
    5. Generate Commentary
    """
    ticker = ticker.upper()
    
    # 1. Load Data
    try:
        prices = dl.load_prices(tickers=ticker)
        technicals = dl.load_technicals(tickers=ticker)
        macro = dl.load_macro()
        fundamentals = dl.load_fundamentals(tickers=ticker)
        
        if prices is None or prices.empty:
            raise HTTPException(status_code=404, detail="No price data found for ticker")
            
        prices = pp.clean_data(prices)
        df = pp.merge_data(prices, technicals, macro, fundamentals)
        
        # Get Latest Row
        latest_row = df.iloc[[-1]].copy()
        latest_date = str(latest_row['date'].iloc[0].date())
        
        # Ensure all model features exist
        for c in FEATURE_COLS:
            if c not in latest_row.columns:
                latest_row[c] = 0.0
            
        # Select Features (Strict Order)
        X_latest = latest_row[FEATURE_COLS].astype(float)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data processing error: {str(e)}")

    # 2. Inference
    try:
        # Explain
        explanation = xai.explain_latest(MODEL, X_latest)
        
        # Add commentary
        commentary = llm.generate_market_commentary(ticker, explanation)
        
        # Structure Response
        response = {
            "ticker": ticker,
            "date": latest_date,
            "prediction": explanation['prediction'],
            "confidence": explanation['confidence'],
            "base_probability": explanation['base_value'],
            "bullish_args": explanation['bullish_args'],
            "bearish_args": explanation['bearish_args'],
            "ai_commentary": commentary,
            "current_price": float(latest_row['close'].iloc[0]) if 'close' in latest_row else 0.0
        }
        
        return sanitize_json(response)

        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
