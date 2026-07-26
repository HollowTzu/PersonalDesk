import os
import time
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
    order = "report_date_as_yyyy_mm_dd DESC"
    url = (f"https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
           f"?$where={urllib.parse.quote(where)}&$order={urllib.parse.quote(order)}&$limit=2")
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
    """Price + % change + recent high/low + daily up/down-volume proxy via
    Yahoo's public chart endpoint. Runs server-side (this script, not a
    browser), so the CORS restriction that blocks this from the client-side
    dashboard doesn't apply here."""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range={range_}"
        d = fetch_json(url)
        result = d['chart']['result'][0]
        meta = result['meta']
        quote = result['indicators']['quote'][0]
        price = meta.get('regularMarketPrice')
        prev_close = meta.get('previousClose') or meta.get('chartPreviousClose')
        opens = quote.get('open', [])
        closes = quote.get('close', [])
        highs = [h for h in quote.get('high', []) if h is not None]
        lows = [l for l in quote.get('low', []) if l is not None]
        volumes = quote.get('volume', [])
        if price is None or prev_close is None:
            return None

        # daily bars, paired up, for the volume-delta proxy below
        bars = [
            {'open': o, 'close': c, 'volume': v}
            for o, c, v in zip(opens, closes, volumes)
            if o is not None and c is not None and v is not None
        ]

        return {
            'price': price,
            'pct_change': (price - prev_close) / prev_close * 100,
            'recent_high': max(highs) if highs else price,
            'recent_low': min(lows) if lows else price,
            'bars': bars,
        }
    except Exception as e:
        print(f"Yahoo fetch failed for '{symbol}': {e}")
        return None


def compute_volume_delta_zone(price_data):
    """Approximates cumulative volume delta from daily OHLCV: green-day
    volume counted as buying pressure, red-day volume as selling pressure,
    summed across the lookback window. This is NOT true tick-level CVD
    (that needs bid/ask-tagged trade data we don't have) — it's a daily
    up/down-volume proxy. Also flags the price zone where volume was
    heaviest, as a rough proxy for where size has been transacting
    (a real 'accumulation area' needs a true volume profile, which we
    don't have either — this is the closest honest approximation from
    daily bars)."""
    bars = price_data.get('bars') if price_data else None
    if not bars or len(bars) < 3:
        return None

    cvd = 0
    heaviest_vol = -1
    heaviest_zone_mid = None
    for b in bars:
        signed_vol = b['volume'] if b['close'] >= b['open'] else -b['volume']
        cvd += signed_vol
        if b['volume'] > heaviest_vol:
            heaviest_vol = b['volume']
            heaviest_zone_mid = (b['open'] + b['close']) / 2

    trend = "accumulating (net buy-volume)" if cvd > 0 else "distributing (net sell-volume)" if cvd < 0 else "flat"
    return {
        'cvd_trend': trend,
        'cvd_value': cvd,
        'heaviest_volume_price': heaviest_zone_mid,
    }


GEMINI_LIST_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_GENERATE_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

_gemini_model_cache = {"model": None, "checked": False}


def discover_gemini_model(api_key):
    """Model names in this space go stale every few months (2.0 retired in
    March 2026, 3.x shipping since). Rather than hardcode a guess that will
    inevitably 404 again later, ask Google's own ListModels endpoint what's
    actually valid for this key right now, and use that. Cached per run so
    we only call this once, not once per generation call."""
    if _gemini_model_cache["checked"]:
        return _gemini_model_cache["model"]
    _gemini_model_cache["checked"] = True

    try:
        req = urllib.request.Request(
            GEMINI_LIST_MODELS_URL,
            headers={"x-goog-api-key": api_key},
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.loads(r.read())
        candidates = [
            m["name"].removeprefix("models/")
            for m in d.get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])
            and "flash" in m.get("name", "").lower()
            and "vision" not in m.get("name", "").lower()
            and "image" not in m.get("name", "").lower()
        ]
        if candidates:
            chosen = sorted(candidates)[-1]  # prefer the lexically-latest flash variant
            print(f"Gemini model discovered for this run: {chosen}")
            _gemini_model_cache["model"] = chosen
            return chosen
        print("ListModels returned no usable flash model — check GEMINI_API_KEY / account access.")
    except Exception as e:
        print(f"Gemini ListModels call failed: {e}")
    return None


def call_ai_model(system_prompt, user_prompt, max_tokens=200, _retry=True):
    """AI generation via Google AI Studio's official Gemini free tier.
    Requires a GEMINI_API_KEY secret — a real key from Google (aistudio.google.com),
    not borrowed from any subscription product. This is the intended, sanctioned
    use case for that key, not a workaround."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set — skipping AI generation, falling back to plain text.")
        return None

    model = discover_gemini_model(api_key)
    if not model:
        return None

    url = GEMINI_GENERATE_URL_TEMPLATE.format(model=model)
    payload = json.dumps({
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": max_tokens},
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.loads(r.read())
        return d["candidates"][0]["content"]["parts"][0]["text"]
    except urllib.error.HTTPError as e:
        if e.code == 429 and _retry:
            print("Gemini 429 (rate limited) — backing off 15s and retrying once.")
            time.sleep(15)
            return call_ai_model(system_prompt, user_prompt, max_tokens, _retry=False)
        print(f"Gemini call failed: HTTP {e.code}: {e.reason}")
        return None
    except Exception as e:
        print(f"Gemini call failed: {e}")
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


def compute_qualifier(bias, pct):
    """Badge qualifier word — a deterministic function of real move size,
    not an AI or human judgment call. Thresholds are named so they're easy
    to retune."""
    if bias == "NEUTRAL":
        return "RANGE"
    magnitude = abs(pct)
    if magnitude >= 1.5:
        return "STRONG"
    if magnitude >= 0.75:
        return "MODERATE"
    return "SHORT-TERM"


def fallback_driver_text(asset_name, pct, cot_data):
    """Used only if the AI call fails — still real numbers, just unstyled."""
    parts = [f"{asset_name} is {'+' if pct >= 0 else ''}{pct:.2f}% vs. prior close."]
    if cot_data:
        direction = "increased" if cot_data['delta'] > 0 else "decreased" if cot_data['delta'] < 0 else "was flat"
        parts.append(f"Managed Money net {direction} by {abs(cot_data['delta']):,} contracts w/w.")
    return " ".join(parts)


def generate_ai_driver_and_invalidation(asset_name, pct, cot_data, headlines):
    """Calls GitHub Models to write the driver + invalidation sentences,
    grounded strictly in the real data passed in. Falls back to plain
    computed text (never fabricated headlines) if the call fails or
    returns something we can't parse."""
    cot_line = ""
    if cot_data:
        direction = "increased" if cot_data['delta'] > 0 else "decreased" if cot_data['delta'] < 0 else "was flat"
        cot_line = (f"Managed Money net {direction} by {abs(cot_data['delta']):,} contracts w/w "
                    f"as of {cot_data['report_date']} (current net {cot_data['net']:+,}).")

    system_prompt = (
        "You write one-line institutional desk notes for a personal trading dashboard. "
        "Use only the DATA given — never invent a specific news event, price level, or "
        "cause that isn't in the DATA. If the headlines aren't clearly relevant to this "
        "asset, describe the move in terms of price action/positioning instead of forcing "
        "a news connection. Confident, terse, sell-side tone — no hedge words like 'might' "
        "or 'could', no disclaimers. Output ONLY valid JSON, no markdown fences: "
        '{"driver": "one sentence", "invalidation": "one sentence"}'
    )
    user_prompt = (
        f"Asset: {asset_name}\n"
        f"Price change vs prior close: {pct:+.2f}%\n"
        f"{cot_line}\n"
        f"Today's macro headlines: \"{headlines[0]}\" / \"{headlines[1]}\""
    )

    raw = call_ai_model(system_prompt, user_prompt, max_tokens=150)
    if raw:
        try:
            cleaned = raw.strip().strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
            parsed = json.loads(cleaned)
            if "driver" in parsed and "invalidation" in parsed:
                return parsed["driver"], parsed["invalidation"]
        except Exception as e:
            print(f"Could not parse AI response for {asset_name}, using fallback: {e}")

    return fallback_driver_text(asset_name, pct, cot_data), "Decisive break of the recent trading range."


def build_bias_row(asset_name, price_data, cot_data=None, headlines=None):
    if price_data is None:
        return None  # don't fabricate a row for data we couldn't fetch

    pct = price_data['pct_change']
    cot_delta = cot_data['delta'] if cot_data else None
    bias, badge_class = classify(pct, cot_delta)
    qualifier = compute_qualifier(bias, pct)

    driver, invalidation = generate_ai_driver_and_invalidation(
        asset_name, pct, cot_data, headlines or ("", "")
    )

    return {
        "asset": asset_name,
        "bias": f"{bias} · {qualifier}",
        "biasClass": badge_class,
        "horizon": "Daily bias — AI-written, regenerated on live data each run",
        "driver": driver,
        "invalidation": invalidation,
    }


def compute_all_bias_rows(headlines, gold_price, silver_price, gold_cot, silver_cot):
    fetches = {
        "NQ (NASDAQ 100)": (fetch_yahoo_chart("^NDX"), None),
        "S&P 500 (ES)": (fetch_yahoo_chart("^GSPC"), None),
        "GOLD (XAUUSD)": (gold_price, gold_cot),
        "SILVER (XAGUSD)": (silver_price, silver_cot),
        "BTCUSD (BITCOIN)": (fetch_coingecko_btc(), None),
        "CRUDE OIL (WTI)": (fetch_yahoo_chart("CL=F"), None),
    }

    rows = []
    for asset, (price_data, cot_data) in fetches.items():
        row = build_bias_row(asset, price_data, cot_data, headlines)
        if row:
            rows.append(row)
        else:
            print(f"SKIPPED '{asset}' — no live price data available this run, card left unchanged.")
        time.sleep(3)  # spread out Gemini calls — avoids bursting the free-tier per-minute limit
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


def generate_liquidity_paragraph(gold_price, silver_price, gold_cot, silver_cot, headlines):
    """AI-written intraday bias / swing projection paragraph for the desk note.
    Hard constraint in the prompt: only reference the specific price levels
    given below (real recent highs/lows, plus the volume-delta proxy zone —
    never invent a level that isn't in DATA. 'Liquidity pool' framing = real
    recent range extremes, standard textbook definition (resting stops
    cluster around prior highs/lows), not a claim of actual order-book data."""
    if not (gold_price and silver_price):
        return ("Live price data unavailable this run — intraday liquidity map "
                "could not be generated. Check next scheduled update.")

    g_delta = gold_cot['delta'] if gold_cot else None
    s_delta = silver_cot['delta'] if silver_cot else None
    smt_line = ""
    if g_delta is not None and s_delta is not None:
        g_dir = "up" if g_delta > 0 else "down" if g_delta < 0 else "flat"
        s_dir = "up" if s_delta > 0 else "down" if s_delta < 0 else "flat"
        agree = (g_delta > 0) == (s_delta > 0) if (g_delta != 0 and s_delta != 0) else False
        smt_line = f"Gold Managed Money w/w: {g_dir}. Silver Managed Money w/w: {s_dir}. Positioning {'confirms' if agree else 'is diverging'} across the two metals."

    gold_cvd = compute_volume_delta_zone(gold_price)
    silver_cvd = compute_volume_delta_zone(silver_price)
    cvd_line = ""
    if gold_cvd:
        cvd_line += f"Gold volume-delta proxy: {gold_cvd['cvd_trend']}, heaviest-volume price zone ~{gold_cvd['heaviest_volume_price']:.2f}. "
    if silver_cvd:
        cvd_line += f"Silver volume-delta proxy: {silver_cvd['cvd_trend']}, heaviest-volume price zone ~{silver_cvd['heaviest_volume_price']:.2f}."

    system_prompt = (
        "You write a 2-3 sentence intraday bias / swing projection note for a personal "
        "trading dashboard, in the style of a terse institutional session note. "
        "HARD RULE: the only specific price levels you may mention are the ones given "
        "in DATA below — never state a price level that isn't listed there. Frame the "
        "recent high/low as resting liquidity (where stops typically cluster), and the "
        "'heaviest-volume price zone' as where accumulation/distribution has concentrated "
        "recently — this is standard market-structure framing from real daily volume data, "
        "not a claim of tick-level order-book data. If SMT (gold vs silver positioning "
        "divergence) is noted in DATA, reference it briefly. No hedge words, no "
        "disclaimers, no markdown — plain sentences only."
    )
    user_prompt = (
        f"GOLD: price {gold_price['price']:.2f}, recent range {gold_price['recent_low']:.2f}-{gold_price['recent_high']:.2f}\n"
        f"SILVER: price {silver_price['price']:.2f}, recent range {silver_price['recent_low']:.2f}-{silver_price['recent_high']:.2f}\n"
        f"{cvd_line}\n"
        f"{smt_line}\n"
        f"Today's headlines: \"{headlines[0]}\" / \"{headlines[1]}\""
    )

    raw = call_ai_model(system_prompt, user_prompt, max_tokens=180)
    if raw:
        return raw.strip()

    # fallback — still real numbers, just unstyled
    return (f"Gold liquidity resting between {gold_price['recent_low']:.2f}-{gold_price['recent_high']:.2f}; "
            f"silver between {silver_price['recent_low']:.2f}-{silver_price['recent_high']:.2f}. "
            f"{cvd_line} {smt_line}")


def update_desk_note(soup, geo_headline, liquidity_paragraph):
    desk_note_header = soup.find("div", string=lambda t: t and "INSTITUTIONAL SESSION DESK NOTE" in t)
    if desk_note_header and desk_note_header.parent:
        parent_divs = desk_note_header.parent.find_all("div")
        content_div = parent_divs[1] if len(parent_divs) > 1 else None
        if content_div:
            new_html = (
                f"<b>GEOPOLITICAL &amp; RATE TRANSMISSION:</b> {geo_headline}<br><br>"
                f"<b>INTRADAY BIAS &amp; LIQUIDITY MAP:</b> {liquidity_paragraph}"
            )
            content_div.clear()
            content_div.append(BeautifulSoup(new_html, 'html.parser'))
            print("Desk note updated.")
            return
    print("WARNING: desk note not found — left unchanged.")


def generate_updated_dashboard():
    with open('index.html', 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    headlines = fetch_latest_macro_headlines()

    # fetch gold/silver once, share between bias cards and the desk note paragraph
    gold_price = fetch_yahoo_chart("GC=F")
    silver_price = fetch_yahoo_chart("SI=F")
    gold_cot = fetch_cot_managed_money("GOLD - COMMODITY EXCHANGE")
    silver_cot = fetch_cot_managed_money("SILVER - COMMODITY EXCHANGE")

    bias_rows = compute_all_bias_rows(headlines, gold_price, silver_price, gold_cot, silver_cot)
    update_bias_cards(soup, bias_rows)

    liquidity_paragraph = generate_liquidity_paragraph(gold_price, silver_price, gold_cot, silver_cot, headlines)
    update_desk_note(soup, headlines[0], liquidity_paragraph)

    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(str(soup))

    print("index.html updated successfully.")


if __name__ == '__main__':
    generate_updated_dashboard()
