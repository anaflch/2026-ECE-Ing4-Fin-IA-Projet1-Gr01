import os
import requests
from dotenv import load_dotenv

from pathlib import Path

# Load environment variables
env_path = Path(__file__).parent.parent / '.env.local'
load_dotenv(env_path)

API_KEY = os.getenv('NEXT_PUBLIC_SUPABASE_ANON_KEY')
BASE_URL = os.getenv('NEXT_PUBLIC_SUPABASE_URL')

if not BASE_URL:
    print("❌ Error: NEXT_PUBLIC_SUPABASE_URL not found in .env.local")
    exit(1)

headers = {
    "apikey": API_KEY,
    "Authorization": f"Bearer {API_KEY}"
}

def check_table(table_name):
    # Supabase REST endpoint pattern: BASE_URL/rest/v1/table_name
    url = f"{BASE_URL}/rest/v1/{table_name}?select=*&limit=1"
    print(f"Checking {table_name}...")
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data:
                print(f"✅ Table '{table_name}' exists and has data. Columns: {list(data[0].keys())}")
                return True
            else:
                print(f"⚠️ Table '{table_name}' exists but is empty.")
                return True
        else:
            print(f"❌ Failed to access '{table_name}'. Status: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

# Common names for fundamentals
# Common names for fundamentals
# Try exact names and variations
check_table("fundamentals_serving")
check_table("Fundamentals_serving")
check_table("fundamentals_raw")
check_table("Fundamentals_raw")


