import os
import urllib.request
import urllib.error
import urllib.parse
import json
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) MarketDeskBot/1.0'}


# ---------------------------------------------------------------------------
# DATA FETCHERS — each returns None on failure rather than raising, so one
# dead source never takes down the whole run. Every number that ends up in
# the dashboard traces back to one of these calls — nothing below this
# point is invented.
# ---------------------------------------------------------------------------

def fetch_json(url, timeout=15):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def fetch_cot_managed_money(search_term):
    """Latest + prior week Managed Money net position from CFTC (Disaggregated Futures-Only)."""
    where = f"upper(market_and_exchange_names) like upper('%{search_term}%')"
    url = (f"https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
           f"?$where={urllib.parse.quote(where)}&$order=report_date_as_yyyy_mm_dd DESC&$limit=2")
    try:
        rows = fetch_json(url)
        if len(rows) < 2:
            return None
        latest, prior = rows[0], rows[1]
        latest_net = int(latest['m_money_positions_long_all']) - int(latest['m_money_positions_short_all'])
        prior_net = int(prior['m_money_positions_long_all']) - int(prior['m_money_positions_short_all'])
        return {
            'net': latest_net,
            'delta': latest_net - prior_net,
            'report_date': latest['report_date_as_yyyy_mm_dd'][:10]
        }
    except Exception as e:
        print(f"COT fetch failed for '{search_term}': {e}")
        return None


def fetch_coingecko_btc():
    try:
        d = fetch_json("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true")
        return {'price': d['bitcoin']['usd'], 'pct_change': d['bitcoin']['usd_24h_change']}
    except Exception as e:
        print(f"CoinGecko fetch failed: {e}")
        return None


def fetch_yahoo_chart(symbol, range_="10d"):
    """Price + % change + recent high/low via Yahoo's public chart endpoint.
    Runs server-side (this script, not a browser), so the CORS restriction
    that blocks this from the client-side dashboard doesn't apply here."""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range={range_}"
        d = fetch_json(url)
        result = d['chart']['result'][0]
        meta = result['meta']
        price = meta.get('regularMarketPrice')
        prev_close = meta.get('previousClose') or meta.get('chartPreviousClose')
        closes = [c for c in result['indicators']['quote'][0]['close'] if c is not None]
        highs = [h for h in result['indicators']['quote'][0]['high'] if h is not None]
        lows = [l for l in result['indicators']['quote'][0]['low'] if l is not None]
        if price is None or prev_close is None:
            return None
        return {
            'price': price,
            'pct_change': (price - prev_close) / prev_close * 100,
            'recent_high': max(highs) if highs else price,
            'recent_low': min(lows) if lows else price,
        }
    except Exception as e:
        print(f"Yahoo fetch failed for '{symbol}': {e}")
        return None


def fetch_te_calendar_highlight(api_key):
    """Optional enrichment: most relevant upcoming high-importance event from
    Trading Economics. Only runs if TE_API_KEY is set as a GitHub Secret —
    never hardcode a key directly in this file."""
    if not api_key:
        return None
    try:
        url = f"https://api.tradingeconomics.com/calendar?c={api_key}&importance=3"
        events = fetch_json(url, timeout=15)
        if events:
            e = events[0]
            return f"{e.get('Event','')} ({e.get('Country','')}) — {e.get('Date','')[:10]}"
    except Exception as e:
        print(f"TradingEconomics fetch failed (check TE_API_KEY secret): {e}")
    return None


def fetch_latest_macro_headlines():
    try:
        req = urllib.request.Request("https://finance.yahoo.com/news/rssindex", headers=UA)
        with urllib.request.urlopen(req, timeout=15) as response:
            xml_data = response.read()
        root = ET.fromstring(xml_data)
        items = root.findall('./channel/item')
        if len(items) >= 2:
            return items[0].find('title').text, items[1].find('title').text
    except Exception as e:
        print(f"Error fetching RSS news: {e}")
    return (
        "Live headline feed unavailable this run — showing last-known macro context.",
        "Precious metals positioning continues to reflect the structural COT trend tracked above."
    )


# ---------------------------------------------------------------------------
# BIAS ENGINE — the actual formula, fully visible. This is a transparent
# heuristic (price momentum, and for gold/silver, whether Managed Money COT
# positioning is confirming or diverging from price) — not a black box, and
# not a claim of predictive edge. Thresholds are named constants so they're
# easy to tune.
# ---------------------------------------------------------------------------

PRICE_MOVE_THRESHOLD = 0.5   # % move to count as a directional lean
COT_AGREEMENT_BONUS = True   # for gold/silver, require COT to agree with price for a strong call


def classify(pct_change, cot_delta=None):
    price_lean = 1 if pct_change >= PRICE_MOVE_THRESHOLD else (-1 if pct_change <= -PRICE_MOVE_THRESHOLD else 0)

    if cot_delta is None:
        if price_lean == 1:
            return "BULLISH", "badge-bullish"
        if price_lean == -1:
            return "BEARISH", "badge-bearish"
        return "NEUTRAL", "badge-neutral"

    cot_lean = 1 if cot_delta > 0 else (-1 if cot_delta < 0 else 0)
    if price_lean == 1 and cot_lean >= 0:
        return "BULLISH", "badge-bullish"
    if price_lean == -1 and cot_lean <= 0:
        return "BEARISH", "badge-bearish"
    return "NEUTRAL", "badge-neutral"  # price and positioning disagree — no clean call


def build_bias_row(asset_name, price_data, cot_data=None, te_note=None):
    if price_data is None:
        return None  # don't fabricate a row for data we couldn't fetch

    pct = price_data['pct_change']
    cot_delta = cot_data['delta'] if cot_data else None
    bias, badge_class = classify(pct, cot_delta)

    driver_parts = [f"{asset_name} is {'+' if pct >= 0 else ''}{pct:.2f}% vs. prior close."]
    if cot_data:
        direction = "increased" if cot_data['delta'] > 0 else "decreased" if cot_data['delta'] < 0 else "was flat"
        driver_parts.append(
            f"Managed Money net {direction} by {abs(cot_data['delta']):,} contracts "
            f"w/w as of {cot_data['report_date']} (net {cot_data['net']:+,})."
        )
        if (pct >= PRICE_MOVE_THRESHOLD and cot_data['delta'] < 0) or (pct <= -PRICE_MOVE_THRESHOLD and cot_data['delta'] > 0):
            driver_parts.append("Price and positioning are diverging — treated as NEUTRAL rather than forcing a call.")
    if te_note:
        driver_parts.append(f"Watch: {te_note}")

    high = price_data.get('recent_high')
    low = price_data.get('recent_low')
    if bias == "BEARISH" and high:
        invalidation = f"Reclaim back above the recent high (~{high:,.2f})."
    elif bias == "BULLISH" and low:
        invalidation = f"Breakdown below the recent low (~{low:,.2f})."
    elif high and low:
        invalidation = f"Decisive break of the recent range (~{low:,.2f}-{high:,.2f})."
    else:
        invalidation = "Decisive break of the recent trading range."

    return {
        "asset": asset_name,
        "bias": bias,
        "biasClass": badge_class,
        "horizon": f"Daily bias — computed from live data, run-updated",
        "driver": " ".join(driver_parts),
        "invalidation": invalidation,
    }


def compute_all_bias_rows():
    te_key = os.environ.get("TE_API_KEY")
    te_note = fetch_te_calendar_highlight(te_key)

    gold_cot = fetch_cot_managed_money("GOLD - COMMODITY EXCHANGE")
    silver_cot = fetch_cot_managed_money("SILVER - COMMODITY EXCHANGE")

    fetches = {
        "NQ (NASDAQ 100)": (fetch_yahoo_chart("^NDX"), None),
        "S&P 500 (ES)": (fetch_yahoo_chart("^GSPC"), None),
        "GOLD (XAUUSD)": (fetch_yahoo_chart("GC=F"), gold_cot),
        "SILVER (XAGUSD)": (fetch_yahoo_chart("SI=F"), silver_cot),
        "BTCUSD (BITCOIN)": (fetch_coingecko_btc(), None),
        "CRUDE OIL (WTI)": (fetch_yahoo_chart("CL=F"), None),
    }

    rows = []
    for asset, (price_data, cot_data) in fetches.items():
        row = build_bias_row(asset, price_data, cot_data, te_note if asset in ("GOLD (XAUUSD)", "SILVER (XAGUSD)") else None)
        if row:
            rows.append(row)
        else:
            print(f"SKIPPED '{asset}' — no live price data available this run, card left unchanged.")
    return rows


# ---------------------------------------------------------------------------
# DOM UPDATE — same targeting logic verified against the real HTML structure.
# ---------------------------------------------------------------------------

def update_bias_cards(soup, bias_rows):
    updated_count = 0
    cards = soup.find_all("div", class_="bias-card")
    by_asset = {row["asset"]: row for row in bias_rows}

    for card in cards:
        asset_span = card.find("span", class_="bias-asset")
        if not asset_span:
            continue
        asset_name = asset_span.get_text(strip=True)
        row = by_asset.get(asset_name)
        if not row:
            print(f"No live data computed for '{asset_name}' this run — left unchanged.")
            continue

        badge = card.find("span", class_="bias-badge")
        if badge:
            badge["class"] = ["bias-badge", row["biasClass"]]
            badge.string = row["bias"]

        meta = card.find("div", class_="bias-meta")
        if meta:
            meta.string = row["horizon"]

        driver = card.find("div", class_="bias-driver")
        if driver:
            driver.string = row["driver"]

        invalidation = card.find("div", class_="bias-invalidation")
        if invalidation:
            invalidation.clear()
            b_tag = soup.new_tag("b")
            b_tag.string = "Flip condition:"
            invalidation.append(b_tag)
            invalidation.append(" " + row["invalidation"])

        updated_count += 1

    print(f"Updated {updated_count} of {len(cards)} bias cards with live data.")


def update_desk_note(soup, geo_headline, supply_headline):
    desk_note_header = soup.find("div", string=lambda t: t and "INSTITUTIONAL SESSION DESK NOTE" in t)
    if desk_note_header and desk_note_header.parent:
        parent_divs = desk_note_header.parent.find_all("div")
        content_div = parent_divs[1] if len(parent_divs) > 1 else None
        if content_div:
            new_html = (
                f"<b>GEOPOLITICAL &amp; RATE TRANSMISSION:</b> {geo_headline}<br><br>"
                f"<b>STRUCTURAL PHYSICAL DEFICIT:</b> {supply_headline}"
            )
            content_div.clear()
            content_div.append(BeautifulSoup(new_html, 'html.parser'))
            print("Desk note updated.")
            return
    print("WARNING: desk note not found — left unchanged.")


def generate_updated_dashboard():
    with open('index.html', 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    bias_rows = compute_all_bias_rows()
    update_bias_cards(soup, bias_rows)

    geo_headline, supply_headline = fetch_latest_macro_headlines()
    update_desk_note(soup, geo_headline, supply_headline)

    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(str(soup))

    print("index.html updated successfully.")


if __name__ == '__main__':
    generate_updated_dashboard()
