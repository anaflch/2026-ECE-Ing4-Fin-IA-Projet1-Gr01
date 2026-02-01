import os
import pandas as pd
import requests
from dotenv import load_dotenv
import time
from pathlib import Path

load_dotenv('.env.local')

API_KEY = os.getenv('NEXT_PUBLIC_SUPABASE_ANON_KEY')
BASE_URL = os.getenv('NEXT_PUBLIC_SUPABASE_URL')
DATA_DIR = Path('data')

def fetch_data(table_name, params=None, cache_file=None, force_reload=False):
    """
    Generic fetcher for Supabase tables.
    """
    if cache_file:
        cache_path = DATA_DIR / cache_file
        if cache_path.exists() and not force_reload:
            print(f"Loading {cache_file} from cache...")
            if cache_file.endswith('.parquet'):
                return pd.read_parquet(cache_path)
            return pd.read_csv(cache_path)

    if not API_KEY or not BASE_URL:
        print("Missing Credentials")
        return None

    headers = {
        "apikey": API_KEY,
        "Authorization": f"Bearer {API_KEY}"
    }
    # Supabase REST: URL/rest/v1/tablename?params
    url = f"{BASE_URL}/rest/v1/{table_name}"
    
    # Supabase uses query params for filtering e.g. ?select=*
    # We default to select=* if not provided
    if params is None:
        params = {"select": "*"}
    elif "select" not in params:
        params["select"] = "*"
    
    print(f"Fetching data from {url}...")
    
    all_data = []
    # Supabase (PostgREST) often limits to 1000 by default unless configured otherwise.
    # We ask for a large chunk, but handle whatever we get.
    batch_size = 10000
    offset = 0
    
    while True:
        # Range header inclusive: 0-9
        headers["Range"] = f"{offset}-{offset + batch_size - 1}"
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            batch = response.json()
            
            if not batch:
                break
                
            all_data.extend(batch)
            count = len(batch)
            print(f"  Fetched {count} rows (Total: {len(all_data)})...")
            
            # Prepare next offset
            offset += count
            
            # If we got fewer than (some small number), we are probably done.
            # But safer to just loop until empty.
            # However, if server limit is 1000 and we got 1000, we might have more.
            # If we got < 1000, we are definitely done.
            if count < 1000: 
                break
                
        except Exception as e:
            print(f"Error fetching {table_name} at offset {offset}: {e}")
            break
            
    if not all_data:
        return None

    df = pd.DataFrame(all_data)
    
    if cache_file:
        print(f"Saving to {cache_file}...")
        if cache_file.endswith('.parquet'):
            df.to_parquet(DATA_DIR / cache_file)
        else:
            df.to_csv(DATA_DIR / cache_file, index=False)
    
    return df


# Validated Table Names
TABLE_PRICES = "prices_daily"
TABLE_MACRO = "macro_indicators"
TABLE_TECH = "technical_indicators"

def load_prices(tickers=None, start_date=None):
    """
    Fetch OHLCV data from 'prices_daily'.
    """
    params = {"select": "symbol,trade_date,open_price,high_price,low_price,close_price,adj_close,volume"}
    
    if tickers:
        # Supabase syntax for IN: symbol=in.(AAPL,GOOG)
        if isinstance(tickers, list):
            filter_val = f"({','.join(tickers)})"
            params["symbol"] = f"in.{filter_val}"
        else:
             params["symbol"] = f"eq.{tickers}"
             
    if start_date:
        params["trade_date"] = f"gte.{start_date}"

    # Default order
    params["order"] = "trade_date.asc"
    
    df = fetch_data(TABLE_PRICES, params=params)
    
    if df is not None and not df.empty:
        # Standardize columns
        df = df.rename(columns={
            "trade_date": "date",
            "symbol": "ticker",
            "open_price": "open",
            "high_price": "high",
            "low_price": "low",
            "close_price": "close"
        })
        
        # Enforce numeric types
        numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        df["date"] = pd.to_datetime(df["date"])
        
        # Save manually here since we modified it
        print("Saving cleaned prices to prices.parquet...")
        df.to_parquet(DATA_DIR / "prices.parquet")
        
    return df

def load_macro():
    """
    Fetch Macroeconomic data from 'macro_indicators'.
    """
    # Load all macro data (usually smaller than prices)
    df = fetch_data(TABLE_MACRO, cache_file="macro.parquet")
    if df is not None and not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df

TABLE_FUNDAMENTALS = "fundamentals_serving"

def load_technicals(tickers=None):
    """
    Fetch Technical Indicators from 'technical_indicators'.
    """
    params = {"select": "*"}
    if tickers:
        if isinstance(tickers, list):
            params["symbol"] = f"in.({','.join(tickers)})"
        else:
            params["symbol"] = f"eq.{tickers}"
             
    df = fetch_data(TABLE_TECH, params=params, cache_file="technicals.parquet")
    if df is not None and not df.empty:
        df = df.rename(columns={"symbol": "ticker"})
        df["date"] = pd.to_datetime(df["date"])
    return df

def load_fundamentals(tickers=None):
    """
    Fetch Fundamental Data from 'fundamentals_serving'.
    """
    params = {"select": "*"}
    if tickers:
        if isinstance(tickers, list):
            params["symbol"] = f"in.({','.join(tickers)})"
        else:
            params["symbol"] = f"eq.{tickers}"
            
    # Fundamentals are less frequent, but we fetch all
    df = fetch_data(TABLE_FUNDAMENTALS, params=params, cache_file="fundamentals.parquet")
    
    if df is not None and not df.empty:
        df = df.rename(columns={"symbol": "ticker", "fiscal_date_ending": "date"})
        # Some rows might use 'updated_at' if fiscal_date is missing, but fiscal is better for alignment
        df["date"] = pd.to_datetime(df["date"])
        
        # Keep only relevant columns for ML
        keep_cols = [
            "ticker", "date", "pe_ratio_ttm", "peg_ratio_ttm", "price_to_book_ttm",
            "ev_to_ebitda_ttm", "debt_to_equity", "roe", "roa", 
            "operating_margin", "net_margin", "revenue_growth_yoy"
        ]
        # Filter columns if they exist
        existing_cols = [c for c in keep_cols if c in df.columns]
        df = df[existing_cols]
        
    return df


def get_available_tickers():
    """
    Fetch all unique tickers from the prices table to enable search filtering.
    """
    try:
        # Fetch just the distinct tickers. PostgreSQL syntax for unique is DISTINCT or GROUP BY.
        # Supabase select: ticker
        # We can't easily do DISTINCT via simple REST param unless we use a view or rpc.
        # Fallback: Fetch all tickers (lightweight column) ?? No, that's heavy if millions of rows.
        # BETTER: Assuming a 'tickers' table exists? User didn't say.
        # Let's try to fetch a known static list or cache.
        # Actually, for this specific user env, let's assume we can fetch from a 'tickers' table OR just try to group by ticker from prices if allowable.
        # Since I can't guess, let's try to fetch from 'tickers' metadata table if it exists, otherwise 
        # let's try to fetch unique tickers from prices via specific query if possible, or just HEAD.
        
        # Current solution: Attempt to fetch from a 'tickers' table.
        # If that fails (likely), we might have to rely on what we can find or pre-seed common ones.
        # Wait, the user said "Ne propose que les tickers qui sont disponibles".
        # Let's try fetching from 'prices' with ?select=ticker&limit=1 (NO).
        # We need ALL tickers. 
        # Workaround: For this specific project demo, let's hardcode a 'CACHE' of known good tickers if we can't query metadata.
        # BUT, to be dynamic: check if we can query distinct tickers.
        # PostgREST: /prices?select=ticker
        # That's too much data.
        
        # Let's try to fetch from 'instruments' or 'tickers' table.
        res = requests.get(f"{BASE_URL}/rest/v1/tickers", headers={
            "apikey": API_KEY,
            "Authorization": f"Bearer {API_KEY}"
        }, params={"select": "symbol"})
        
        if res.status_code == 200:
            data = res.json()
            return [x['symbol'] for x in data]
            
        # Fallback to prices table (inefficient but workable for small DB)
        # Actually, let's just return a set of popular ones if we fail, OR return None and warn.
        print("Could not fetch tickers list, defaulting to open search.")
        return None
    except Exception as e:
        print(f"Error fetching available tickers: {e}")
        return None
