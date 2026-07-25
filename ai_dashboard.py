import json
import os
import requests
from bs4 import BeautifulSoup
from typing import Dict, Tuple, List

# -------------------------------------------------------------------
# TradingEconomics Scraper for Live Prices[cite: 3]
# -------------------------------------------------------------------
TE_URLS = {
    "GOLD (XAUUSD)": "https://tradingeconomics.com/commodity/gold",
    "SILVER (XAGUSD)": "https://tradingeconomics.com/commodity/silver",
    "CRUDE OIL (WTI)": "https://tradingeconomics.com/commodity/crude-oil",
    "NQ (NASDAQ 100)": "https://tradingeconomics.com/us/nasdaq-100",
    "S&P 500 (ES)": "https://tradingeconomics.com/us/sp-500",
    "BTCUSD (BITCOIN)": "https://tradingeconomics.com/crypto/bitcoin"
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def fetch_live_prices() -> Dict[str, str]:
    prices = {}
    for name, url in TE_URLS.items():
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                price_elem = soup.find(id="p") or soup.select_one("#p") or soup.select_one("[id='p']")
                if price_elem:
                    val = float(price_elem.text.strip().replace(",", ""))
                    prices[name] = f"${val:,.2f}"
                else:
                    price_table_cell = soup.select_one("table.table td#p") or soup.select_one(".table-heatmap td")
                    if price_table_cell:
                        val = float(price_table_cell.text.strip().replace(",", ""))
                        prices[name] = f"${val:,.2f}"
                    else:
                        prices[name] = "FETCH_FAILED"
            else:
                prices[name] = "FETCH_FAILED"
        except Exception as e:
            print(f"[ERROR] Failed scraping {name} from TradingEconomics: {e}")
            prices[name] = "FETCH_FAILED"
    return prices


# -------------------------------------------------------------------
# Newsfilter / Realtime-NewsAPI Integration
# -------------------------------------------------------------------
def fetch_institutional_headlines() -> List[str]:
    api_key = os.environ.get("NEWSFILTER_API_KEY")
    if not api_key:
        return [
            "Macro markets balance rate transmission adjustments against persistent geopolitical premiums.",
            "Structural physical deficits remain unhedged across precious metals and key tech components."
        ]

    url = "https://api.newsfilter.io/public/actions"
    payload = {
        "queryString": "types:news AND (category:macro OR category:commodities OR market:US)",
        "from": 0,
        "size": 3
    }
    headers = {
        "Content-Type": "application/json",
        "apiKey": api_key
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])
            headlines = [art.get("title") for art in articles if art.get("title")]
            if headlines:
                return headlines
    except Exception as e:
        print(f"[ERROR] Failed fetching Newsfilter headlines: {e}")

    return [
        "Live feed fallback: Central bank liquidity and sticky energy components drive volatility.",
        "Industrial supply shortfalls continue to support floor positioning across physical assets."
    ]


# -------------------------------------------------------------------
# GitHub Models AI Desk Note Synthesis
# -------------------------------------------------------------------
def generate_ai_desk_note(headlines: List[str]) -> str:
    """
    Passes live headlines into GitHub's built-in Models endpoint 
    (OpenAI-compatible) using GITHUB_TOKEN.
    """
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        return (
            "<b>GEOPOLITICAL &amp; RATE TRANSMISSION:</b> GITHUB_TOKEN missing. "
            f"Latest wire: {headlines[0]}<br><br>"
            "<b>STRUCTURAL PHYSICAL DEFICIT:</b> Core structural supply constraints remain fully intact."
        )

    prompt = f"""
    You are an institutional trading desk macro strategist.
    Analyze these live institutional news feeds:
    1. {headlines[0] if len(headlines) > 0 else 'N/A'}
    2. {headlines[1] if len(headlines) > 1 else 'N/A'}

    Write the "INSTITUTIONAL SESSION DESK NOTE — MACRO REGIME" section with two clear bolded paragraphs:
    - GEOPOLITICAL & RATE TRANSMISSION: Focus on rate expectations, real yields, and geopolitical variables.
    - STRUCTURAL PHYSICAL DEFICIT: Focus on physical supply shortages, industrial demand, or tech capital deployment.
    
    Keep the tone concise, highly professional, and institutional. Return output strictly as two HTML paragraphs with tags.
    """

    payload = {
        "model": "gpt-4o",  # Or another model available in GitHub Models
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }

    try:
        res = requests.post(
            "https://models.inference.ai.azure.com/chat/completions",
            json=payload,
            headers={
                "Authorization": f"Bearer {github_token}",
                "Content-Type": "application/json"
            },
            timeout=20
        )
        if res.status_code == 200:
            return res.json()['choices'][0]['message']['content'].strip()
        else:
            print(f"[ERROR] GitHub Models API error: {res.status_code} - {res.text}")
    except Exception as e:
        print(f"[ERROR] GitHub Models generation failed: {e}")

    return "<b>GEOPOLITICAL &amp; RATE TRANSMISSION:</b> API synthesis error.<br><br><b>STRUCTURAL PHYSICAL DEFICIT:</b> Parameters holding steady."


# -------------------------------------------------------------------
# DOM Injection & Execution Runner
# -------------------------------------------------------------------
def update_html_dashboard():
    with open('index.html', 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    prices = fetch_live_prices()
    headlines = fetch_institutional_headlines()
    ai_note_html = generate_ai_desk_note(headlines)

    desk_note_header = soup.find("div", string=lambda t: t and "INSTITUTIONAL SESSION DESK NOTE" in t)
    if desk_note_header and desk_note_header.parent:
        parent_divs = desk_note_header.parent.find_all("div")
        content_div = parent_divs[1] if len(parent_divs) > 1 else None
        if content_div:
            content_div.clear()
            content_div.append(BeautifulSoup(ai_note_html, 'html.parser'))
            print("[SUCCESS] Institutional session desk note updated via GitHub Models.")

    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(str(soup))

if __name__ == "__main__":
    update_html_dashboard()
