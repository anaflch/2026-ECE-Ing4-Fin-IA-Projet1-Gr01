import os
import requests
from dotenv import load_dotenv
from pathlib import Path

# Load env
env_path = Path(__file__).resolve().parent.parent / '.env.local'
load_dotenv(env_path)

api_url = os.getenv('NEXT_PUBLIC_SUPABASE_URL')
api_key = os.getenv('NEXT_PUBLIC_SUPABASE_ANON_KEY')

headers = {
    "apikey": api_key,
    "Authorization": f"Bearer {api_key}"
}

inspect_tables = ["technical_indicators", "macro_indicators", "fundamentals_raw"]


print(f"\n🔬 Inspecting found tables...")

for table in inspect_tables:
    url = f"{api_url}/rest/v1/{table}?select=*&limit=1"
    try:
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            data = r.json()
            if len(data) > 0:
                print(f"✅ {table} Columns: {list(data[0].keys())}")
            else:
                print(f"✅ {table} exists but is EMPTY.")
        else:
            print(f"❌ {table} not accessible: {r.status_code}")
    except Exception as e:
        print(f"Error inspecting {table}: {e}")

print("\n🕵️ Probing for PRICES table (Phase 2)...")
price_candidates = [
    "prices_daily", "daily_adj_price", "market_quotes", "historical_data", 
    "ohlcv_data", "candle_data", "equities", "stocks_data", "pricing"
]

for table in price_candidates:
    url = f"{api_url}/rest/v1/{table}?select=*&limit=1"
    try:
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
             print(f"✅ FOUND PRICE TABLE: '{table}'")
             if len(r.json()) > 0:
                 print(f"   Columns: {list(r.json()[0].keys())}")
    except:
        pass

