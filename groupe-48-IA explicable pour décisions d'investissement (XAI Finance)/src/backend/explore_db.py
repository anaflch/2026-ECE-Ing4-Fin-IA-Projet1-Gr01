import os
import requests
import json
from dotenv import load_dotenv
from pathlib import Path

# Load env
env_path = Path(__file__).resolve().parent.parent / '.env.local'
load_dotenv(env_path)

api_url = os.getenv('NEXT_PUBLIC_SUPABASE_URL')
api_key = os.getenv('NEXT_PUBLIC_SUPABASE_ANON_KEY')

if not api_url or not api_key:
    print("❌ Credentials missing.")
    exit(1)

print(f"🔍 Analyzing Supabase Schema at {api_url}...")

# Supabase via PostgREST exposes OpenAPI spec at root if we send the key
url = f"{api_url}/rest/v1/"
headers = {
    "apikey": api_key,
    "Authorization": f"Bearer {api_key}",
    "Accept": "application/openapi+json, application/json"
}

try:
    response = requests.get(url, headers=headers)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("\n📊 Found Tables (Definitions):")
        definitions = data.get('definitions', {})
        for table_name in definitions.keys():
            print(f"  - {table_name}")
            
            # Optional: Print properties/columns
            # props = definitions[table_name].get('properties', {})
            # print(f"    Columns: {list(props.keys())}")
            
    else:
        print("❌ Failed to fetch schema.")
        print(response.text[:500])

except Exception as e:
    print(f"❌ Error: {e}")
