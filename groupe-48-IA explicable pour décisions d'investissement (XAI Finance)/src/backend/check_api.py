import os
import requests
from dotenv import load_dotenv
import sys

# Load environment variables
from pathlib import Path
env_path = Path(__file__).resolve().parent.parent / '.env.local'
print(f"Loading env from {env_path}")
load_dotenv(env_path)

api_url = os.getenv('NEXT_PUBLIC_SUPABASE_URL')
api_key = os.getenv('NEXT_PUBLIC_SUPABASE_ANON_KEY')

print(f"Checking API connection...")

if not api_url:
    print("❌ Error: NEXT_PUBLIC_SUPABASE_URL is missing")
    sys.exit(1)

if not api_key:
    print("❌ Error: NEXT_PUBLIC_SUPABASE_ANON_KEY is missing")
    sys.exit(1)

print(f"✅ Credentials found.")
print(f"Target URL: {api_url}")

# Supabase connectivity check
# Try to list a table, or just root if not sure. 
# Usually Supabase REST API is at hostname/rest/v1/
try:
    # We don't know the table names yet, but we can try to hit the root of rest v1
    # or just the base url to see if it replies (Supabase often gives 404 on root but its alive)
    target = f"{api_url}/rest/v1/"
    headers = {
        "apikey": api_key,
        "Authorization": f"Bearer {api_key}"
    }
    response = requests.get(target, headers=headers, timeout=10)
    
    print(f"Status Code: {response.status_code}")
    # Supabase root rest/v1 might return doc or something, or 404 if no tables listed ?
    # Actually usually it lists definitions if Swagger is on, but let's assume if we get a response it's roughly ok.
    # 200 is great. 401 is bad key. 404 ok(ish).
    if response.status_code in [200, 404]: 
        print("✅ Connection appears valid (Server responded).")
    elif response.status_code == 401:
        print("❌ Authorization failed (401). Check Key.")
    else:
        print(f"⚠️ Unexpected status: {response.status_code}")
        print(response.text)

    # Try specific table 'prices' just in case root is 404/401 but tables work
    print("Trying to fetch 1 row from 'prices'...")
    prices_url = f"{api_url}/rest/v1/prices?select=*&limit=1"
    resp_prices = requests.get(prices_url, headers=headers)
    print(f"Prices Status: {resp_prices.status_code}")
    if resp_prices.status_code == 200:
        print("✅ Access to 'prices' table confirmed.")
    else:
        print(f"❌ Failed to acccess 'prices': {resp_prices.status_code} - {resp_prices.text[:100]}")

except Exception as e:
    print(f"❌ Connection failed: {str(e)}")
    sys.exit(1)
