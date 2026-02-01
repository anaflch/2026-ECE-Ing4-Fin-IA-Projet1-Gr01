
import os
import requests
from dotenv import load_dotenv

load_dotenv('.env.local')

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def generate_market_commentary(ticker, explanation):
    """
    Generate a natural language summary of the market analysis using an LLM.
    
    Args:
        ticker (str): Ticker symbol.
        explanation (dict): The output from explainability.explain_latest()
        
    Returns:
        str: Generated text.
    """
    
    # Construct Prompts
    bull_text = ", ".join([f"{a['text']} (Impact: {a['shap']:.2f})" for a in explanation['bullish_args'][:3]])
    bear_text = ", ".join([f"{a['text']} (Impact: {a['shap']:.2f})" for a in explanation['bearish_args'][:3]])
    
    prompt = f"""
    You are a senior financial analyst. 
    Write a concise, 3-sentence professional market commentary for {ticker} based on the following machine learning analysis:
    
    Prediction: {explanation['prediction']} (Confidence: {explanation['confidence']:.1%})
    
    Key Bullish Drivers:
    {bull_text}
    
    Key Bearish Risks:
    {bear_text}
    
    Verdict Style: Professional, objective, yet decisive. 
    Focus on the "why" - explaining the conflict between the factors.
    Avoid standard disclaimers.
    """
    
    # Try Anthropic (Claude) first
    if ANTHROPIC_API_KEY:
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 300,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=10
            )
            data = response.json()
            if 'content' in data:
                return data['content'][0]['text']
            else:
                return f"Error from Claude: {data}"
        except Exception as e:
            print(f"Claude API Error: {e}")
            
    # Fallback to OpenAI (GPT)
    if OPENAI_API_KEY:
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": "You are a financial analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 150
                },
                timeout=10
            )
            data = response.json()
            if 'choices' in data:
                return data['choices'][0]['message']['content']
        except Exception as e:
            print(f"OpenAI API Error: {e}")
            
    # Fallback to Local Rule-Based Generation
    print("⚠️ No API Key found. Using Rule-Based Fallback.")
    
    direction = "Bullish" if explanation['prediction'] == "LONG" else "Bearish"
    confidence_str = "strong" if explanation['confidence'] > 0.6 else "moderate"
    
    # Pick top reason
    primary_reason = "technical factors"
    if explanation['bullish_args'] and direction == "Bullish":
        primary_reason = explanation['bullish_args'][0]['text']
    elif explanation['bearish_args'] and direction == "Bearish":
        primary_reason = explanation['bearish_args'][0]['text']
        
    return f" [AI ANALYST (LOCAL)] The model forecasts a {confidence_str} {direction} trend for {ticker}. This is primarily driven by {primary_reason}. Traders should monitor for confirmation of this signal given the current market regime."
