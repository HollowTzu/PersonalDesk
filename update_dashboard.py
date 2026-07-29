import os
import time
import urllib.request
import urllib.error
import urllib.parse
import json
from datetime import datetime
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup, Comment

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
        excluded_terms = ["vision", "image", "preview", "experimental", "exp", "omni", "audio", "live", "thinking"]
        all_flash = [
            m["name"].removeprefix("models/")
            for m in d.get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])
            and "flash" in m.get("name", "").lower()
        ]
        # prefer stable, non-preview models — preview/experimental/omni variants
        # tend to carry much tighter free-tier rate limits, which is what
        # caused every call to 429 when 'gemini-omni-flash-preview' got picked
        stable = [n for n in all_flash if not any(term in n.lower() for term in excluded_terms)]
        candidates = stable if stable else all_flash
        if not stable and all_flash:
            print("No stable flash model found — falling back to a preview/experimental variant (expect tighter rate limits).")
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
    except (TimeoutError, urllib.error.URLError) as e:
        if _retry:
            print(f"Gemini call timed out — retrying once: {e}")
            return call_ai_model(system_prompt, user_prompt, max_tokens, _retry=False)
        print(f"Gemini call failed after retry: {e}")
        return None
    except Exception as e:
        print(f"Gemini call failed: {e}")
        return None


RSS_SOURCES = [
    ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex"),
    ("MarketWatch", "https://feeds.marketwatch.com/marketwatch/topstories/"),
    ("CNBC", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("FXStreet", "https://www.fxstreet.com/rss"),
]


def fetch_headlines_from(name, url, max_items=4):
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=15) as response:
            xml_data = response.read()
        root = ET.fromstring(xml_data)
        items = root.findall('./channel/item')[:max_items]
        return [i.find('title').text for i in items if i.find('title') is not None]
    except Exception as e:
        print(f"RSS fetch failed for {name}: {e}")
        return []


def fetch_all_headlines():
    """Pull from multiple free financial RSS sources — one dead feed doesn't
    take down the rest. Returns a flat list of real headlines for the AI to
    summarize from (never fabricated, always traceable to a real source)."""
    all_headlines = []
    for name, url in RSS_SOURCES:
        headlines = fetch_headlines_from(name, url)
        all_headlines.extend(headlines)
        print(f"{name}: {len(headlines)} headlines fetched.")
    return all_headlines


def summarize_geopolitical_rate_context(headlines):
    """AI-summarized synthesis across real headlines — replaces the old
    behavior of just using headline[0] verbatim, which was often irrelevant
    (e.g. a random corporate strategy article) since it wasn't filtered for
    actual relevance to rates/geopolitics/precious metals at all."""
    if not headlines:
        return "Live headline feeds unavailable this run — no fresh geopolitical/rate context to summarize."

    system_prompt = (
        "You write a 2-3 sentence institutional desk note summarizing the current "
        "geopolitical and interest-rate backdrop and how it's affecting markets — "
        "gold, the dollar, and broader risk sentiment specifically. Base this ONLY "
        "on the real headlines given below; if none are genuinely relevant to "
        "geopolitics, rates, or macro conditions, say the news flow is quiet on "
        "that front rather than forcing a connection to an unrelated headline. "
        "Confident, terse, sell-side tone. No hedge words, no disclaimers, no markdown."
    )
    user_prompt = "Today's headlines across multiple sources:\n" + "\n".join(f"- {h}" for h in headlines)

    raw = call_ai_model(system_prompt, user_prompt, max_tokens=180)
    if raw:
        return raw.strip()
    # fallback: at least show a real headline rather than nothing, clearly labeled as unsummarized
    return f"(Unsummarized — AI unavailable this run) Recent headline: \"{headlines[0]}\""


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
    headline_sample = "; ".join(f'"{h}"' for h in (headlines or [])[:4]) or "none available this run"
    user_prompt = (
        f"Asset: {asset_name}\n"
        f"Price change vs prior close: {pct:+.2f}%\n"
        f"{cot_line}\n"
        f"Today's macro headlines: {headline_sample}"
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
        asset_name, pct, cot_data, headlines or []
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


def determine_session():
    """Which session this run belongs to, based on real UTC time. Cron is
    fixed-UTC (doesn't auto-adjust for US DST) — worth re-checking the
    cron values in daily_update.yml twice a year around DST changes."""
    hour_utc = datetime.utcnow().hour
    if 23 <= hour_utc or hour_utc < 7:
        return "ASIA"
    elif 7 <= hour_utc < 12:
        return "LONDON"
    else:
        return "NEW YORK"


def fetch_intraday_extremes(symbol):
    """Today's actual high/low so far, from real intraday bars — used to
    objectively check whether a projected level has really been touched."""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=15m&range=1d"
        d = fetch_json(url)
        result = d['chart']['result'][0]
        quote = result['indicators']['quote'][0]
        highs = [h for h in quote.get('high', []) if h is not None]
        lows = [l for l in quote.get('low', []) if l is not None]
        if not highs or not lows:
            return None
        return {"high": max(highs), "low": min(lows)}
    except Exception as e:
        print(f"Intraday extremes fetch failed for {symbol}: {e}")
        return None


def get_or_set_daily_levels(soup, metal_key, price_data, cvd_data, session):
    """At the first run of the day (Asia), project and save today's key
    levels. At later runs (London/NY), read those same saved levels back
    so the note can honestly say what's been touched vs. still resting —
    this is a real persistence + comparison, not the AI recalling anything."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    state_key = f"daily_levels_{metal_key}"
    prior = get_prior_state(soup, state_key)

    if prior is None or prior.get("date") != today or session == "ASIA":
        levels = {
            "date": today,
            "resistance": price_data['recent_high'],
            "support": price_data['recent_low'],
            "volume_zone": cvd_data['heaviest_volume_price'] if cvd_data else None,
        }
        anchor = soup.find(id="monetary-fiscal-section") or soup.body
        set_state_comment(soup, state_key, levels, anchor)
        print(f"{metal_key}: new daily levels projected for {today} at {session} session.")
        return levels
    return prior


def check_levels_touched(levels, intraday_extremes):
    """Objective, computed — not AI judgment."""
    if not levels or not intraday_extremes:
        return {"resistance_touched": None, "support_touched": None}
    return {
        "resistance_touched": intraday_extremes["high"] >= levels["resistance"],
        "support_touched": intraday_extremes["low"] <= levels["support"],
    }


def generate_liquidity_paragraph(gold_price, silver_price, gold_cot, silver_cot, headlines, session, soup):
    """AI-written session note. Structure (in order): the fact that happened,
    a level's role-change (support<->resistance), volume/accumulation
    confirmation, an honest SMT divergence flag, risk-regime context, and a
    specific invalidation condition. Every price level and touched/untouched
    status is computed in Python from real data and handed to the AI —
    never invented."""
    if not (gold_price and silver_price):
        return ("Live price data unavailable this run — intraday liquidity map "
                "could not be generated. Check next scheduled update.")

    gold_cvd = compute_volume_delta_zone(gold_price)
    silver_cvd = compute_volume_delta_zone(silver_price)

    gold_levels = get_or_set_daily_levels(soup, "gold", gold_price, gold_cvd, session)
    silver_levels = get_or_set_daily_levels(soup, "silver", silver_price, silver_cvd, session)

    gold_intraday = fetch_intraday_extremes("GC=F")
    silver_intraday = fetch_intraday_extremes("SI=F")
    gold_touched = check_levels_touched(gold_levels, gold_intraday)
    silver_touched = check_levels_touched(silver_levels, silver_intraday)

    def fmt_touch(label, level, touched):
        if level is None:
            return f"{label}: not available"
        status = "TOUCHED" if touched else "still untouched" if touched is False else "status unknown"
        return f"{label} {level:.2f} — {status}"

    g_delta = gold_cot['delta'] if gold_cot else None
    s_delta = silver_cot['delta'] if silver_cot else None
    smt_line = ""
    if g_delta is not None and s_delta is not None:
        g_dir = "increased" if g_delta > 0 else "decreased" if g_delta < 0 else "was unchanged"
        s_dir = "increased" if s_delta > 0 else "decreased" if s_delta < 0 else "was unchanged"
        agree = (g_delta > 0) == (s_delta > 0) if (g_delta != 0 and s_delta != 0) else False
        smt_line = (f"Gold's Managed Money net position {g_dir} week-over-week; silver's {s_dir}. "
                    f"Positioning {'confirms' if agree else 'is diverging'} across the two metals.")

    system_prompt = (
        "You write a session trading note (Asia/London/New York open) for a personal "
        "trading dashboard. Follow this exact structure, in order:\n"
        "1. Lead with the single most important FACT from DATA (a level touched, or "
        "still holding) — no throat-clearing.\n"
        "2. If a level was touched, state its role change (resistance that's now support, "
        "or vice versa). If untouched, say what that implies about where price is coiling.\n"
        "3. Reference the volume/accumulation zone from DATA — confirm or question whether "
        "it's actually holding.\n"
        "4. State the SMT (gold vs silver positioning) read honestly — if it's diverging, "
        "say so plainly, don't force false confluence.\n"
        "5. One sentence of risk-regime context — supporting, not leading.\n"
        "6. End with ONE specific invalidation condition — a level AND a real condition "
        "(e.g. 'a close back below X, not just a wick').\n"
        "HARD RULE: only reference price levels and touched/untouched status exactly as "
        "given in DATA — never invent one. Confident, terse, sell-side tone. No hedge "
        "words, no disclaimers, no markdown. 4-6 sentences total."
    )
    headline_sample = "; ".join(f'"{h}"' for h in (headlines or [])[:4]) or "none available this run"
    user_prompt = (
        f"SESSION: {session}\n"
        f"GOLD: price {gold_price['price']:.2f}\n"
        f"  {fmt_touch('Resistance', gold_levels.get('resistance'), gold_touched['resistance_touched'])}\n"
        f"  {fmt_touch('Support', gold_levels.get('support'), gold_touched['support_touched'])}\n"
        f"  Volume-accumulation zone: {gold_levels.get('volume_zone')}\n"
        f"SILVER: price {silver_price['price']:.2f}\n"
        f"  {fmt_touch('Resistance', silver_levels.get('resistance'), silver_touched['resistance_touched'])}\n"
        f"  {fmt_touch('Support', silver_levels.get('support'), silver_touched['support_touched'])}\n"
        f"  Volume-accumulation zone: {silver_levels.get('volume_zone')}\n"
        f"{smt_line}\n"
        f"Today's headlines: {headline_sample}"
    )

    raw = call_ai_model(system_prompt, user_prompt, max_tokens=280)
    if raw:
        return raw.strip()

    # fallback — still real numbers, just unstyled
    return (f"[{session}] Gold: resistance {gold_levels.get('resistance')}, support {gold_levels.get('support')}, "
            f"volume zone {gold_levels.get('volume_zone')}. Silver: resistance {silver_levels.get('resistance')}, "
            f"support {silver_levels.get('support')}. {smt_line}")


def update_desk_note(soup, geo_summary, liquidity_paragraph, session):
    desk_note_header = soup.find("div", string=lambda t: t and "INSTITUTIONAL SESSION DESK NOTE" in t)
    if desk_note_header and desk_note_header.parent:
        parent_divs = desk_note_header.parent.find_all("div")
        content_div = parent_divs[1] if len(parent_divs) > 1 else None
        if content_div:
            try:
                from zoneinfo import ZoneInfo
                now_et = datetime.now(ZoneInfo("America/New_York"))
                timestamp = now_et.strftime("%b %d, %Y — %H:%M ET")
            except Exception:
                timestamp = datetime.utcnow().strftime("%b %d, %Y — %H:%M UTC")

            new_html = (
                f"<div style='font-size:9.5px; color:#8a8f88; margin-bottom:6px;'>As of {timestamp} — {session} session</div>"
                f"<b style='color:#9BE7ED;'>GEOPOLITICAL &amp; RATE TRANSMISSION:</b> {geo_summary}<br><br>"
                f"<b style='color:#9BE7ED;'>INTRADAY BIAS &amp; LIQUIDITY MAP:</b> {liquidity_paragraph}"
            )
            content_div.clear()
            content_div.append(BeautifulSoup(new_html, 'html.parser'))
            print("Desk note updated.")
            return
    print("WARNING: desk note not found — left unchanged.")


def fetch_fred_series(series_id, api_key):
    """Generic FRED series fetcher — latest + prior observation."""
    if not api_key:
        return None
    try:
        url = (f"https://api.stlouisfed.org/fred/series/observations"
               f"?series_id={series_id}&api_key={api_key}&file_type=json"
               f"&sort_order=desc&limit=5")
        d = fetch_json(url)
        obs = [o for o in d.get("observations", []) if o.get("value") not in (".", None)]
        if len(obs) < 2:
            return None
        latest, prior = float(obs[0]["value"]), float(obs[1]["value"])
        return {"value": latest, "delta": latest - prior, "date": obs[0]["date"]}
    except Exception as e:
        print(f"FRED fetch failed for {series_id}: {e}")
        return None


def fetch_fred_credit_spread():
    """ICE BofA US High Yield Index Option-Adjusted Spread (BAMLH0A0HYM2) —
    the real, standard credit-spread series, via FRED's official free API."""
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        print("FRED_API_KEY not set — Credit Spreads will show as 'Not Live'.")
        return None
    return fetch_fred_series("BAMLH0A0HYM2", api_key)


def classify_real_yield_regime(real_yield, breakeven):
    """Deterministic classification into one of the three textbook scenarios
    — the AI only writes the narrative, this decides which scenario applies."""
    if real_yield is None or breakeven is None:
        return None
    ry_delta, be_delta = real_yield['delta'], breakeven['delta']
    if ry_delta > 0.01 and abs(be_delta) < 0.02:
        return "tightening"       # nominal up, breakevens flat -> real yield rising -> bearish gold
    if ry_delta <= 0 and be_delta > 0.02:
        return "inflation_fear"   # breakevens outrunning -> neutral/bullish, debasement-hedge signal
    if ry_delta < -0.01 and abs(be_delta) < 0.02:
        return "growth_scare"     # real yield falling on growth fears -> bullish
    return "mixed"


def generate_real_yield_narrative(real_yield, breakeven, scenario, headlines):
    if real_yield is None or breakeven is None:
        return "Live real yield / breakeven data unavailable this run (check FRED_API_KEY)."

    scenario_context = {
        "tightening": "Nominal yields rising while breakevens are flat — real yields are rising because policy is tightening. Textbook bearish-gold mechanics.",
        "inflation_fear": "Breakevens rising faster than nominal yields — inflation fear is outrunning the rate move, so real yields may still be falling. Often neutral-to-bullish, sometimes a genuine debasement-hedge signal.",
        "growth_scare": "Nominal yields falling while breakevens stay flat — real yields falling on growth-scare fears. Bullish: cheaper to hold, plus a safe-haven bid.",
        "mixed": "No single scenario cleanly applies — real yield and breakeven moves are mixed or small.",
    }.get(scenario, "")

    system_prompt = (
        "You write a 2-3 sentence institutional note on what real yields and inflation "
        "breakevens are currently signaling for gold. Use ONLY the DATA and SCENARIO CONTEXT "
        "given — never invent a number not listed. State which of the three textbook scenarios "
        "currently applies and what it implies for gold. Confident, terse, sell-side tone."
    )
    headline_sample = "; ".join(f'"{h}"' for h in (headlines or [])[:3]) or "none available"
    user_prompt = (
        f"10Y real yield (TIPS): {real_yield['value']:.2f}%, changed {real_yield['delta']:+.2f} vs prior reading ({real_yield['date']})\n"
        f"10Y inflation breakeven: {breakeven['value']:.2f}%, changed {breakeven['delta']:+.2f} vs prior reading\n"
        f"SCENARIO CONTEXT: {scenario_context}\n"
        f"Headlines: {headline_sample}"
    )
    raw = call_ai_model(system_prompt, user_prompt, max_tokens=150)
    if raw:
        return raw.strip()
    return f"Real yield {real_yield['value']:.2f}% ({real_yield['delta']:+.2f}), breakeven {breakeven['value']:.2f}% ({breakeven['delta']:+.2f}). {scenario_context}"


def classify_etf_cot_confluence(cot_data, cvd_data):
    """Deterministic 2x2 classification (see the guide page) — AI only
    writes the sentence, this decides which quadrant applies."""
    if cot_data is None or cvd_data is None:
        return None
    cot_rising = cot_data['delta'] > 0
    etf_rising = cvd_data['cvd_value'] > 0
    if cot_rising and etf_rising: return "confluence_bull"
    if cot_rising and not etf_rising: return "fragile_rally"
    if not cot_rising and etf_rising: return "quiet_accumulation"
    return "broad_distribution"


def generate_etf_confluence_narrative(metal, cot_data, cvd_data, quadrant, headlines):
    if cot_data is None or cvd_data is None or quadrant is None:
        return f"{metal} COT/ETF-flow confluence unavailable this run — one or both inputs missing."

    quadrant_context = {
        "confluence_bull": "COT and ETF flow both rising — strongest confluence, broad-based demand.",
        "fragile_rally": "COT rising but ETF flow not — speculative-only rally, fragile, no allocator backing.",
        "quiet_accumulation": "COT falling but ETF flow rising — patient money buying while speculators de-risk, a stealth-strength signal.",
        "broad_distribution": "Both COT and ETF flow falling — broad distribution, weakest environment.",
    }.get(quadrant, "")

    system_prompt = (
        "You write a 1-2 sentence institutional note on COT vs ETF-flow positioning "
        "confluence for one metal. Use ONLY the DATA and CONTEXT given. Confident, terse tone."
    )
    user_prompt = (
        f"Metal: {metal}\n"
        f"COT Managed Money w/w change: {cot_data['delta']:+,} contracts\n"
        f"ETF price/volume-flow proxy: {cvd_data['cvd_trend']}\n"
        f"CONTEXT: {quadrant_context}"
    )
    raw = call_ai_model(system_prompt, user_prompt, max_tokens=100)
    if raw:
        return raw.strip()
    return f"{metal}: {quadrant_context}"


def get_prior_state(soup, key):
    """Generic version of the risk-gauge persistence pattern — reads a named
    state comment so any section can do its own materiality check."""
    comment = soup.find(string=lambda t: isinstance(t, Comment) and f"{key}:" in t)
    if not comment:
        return None
    try:
        return json.loads(comment.split(f"{key}:")[1].strip())
    except Exception:
        return None


def set_state_comment(soup, key, value, anchor_element):
    new_comment = Comment(f" {key}:{json.dumps(value)} ")
    old_comment = soup.find(string=lambda t: isinstance(t, Comment) and f"{key}:" in t)
    if old_comment:
        old_comment.replace_with(new_comment)
    elif anchor_element:
        anchor_element.insert_before(new_comment)


def update_real_yield_section(soup, real_yield, breakeven, scenario, headlines):
    content_el = soup.find(id="real-yield-content")
    if not content_el:
        print("WARNING: real-yield-content element not found — left unchanged.")
        return
    prior = get_prior_state(soup, "real_yield_state")
    current = {"scenario": scenario}
    if prior != current or prior is None:
        print(f"Real yield regime changed ({prior} -> {current}) — regenerating via AI.")
        narrative = generate_real_yield_narrative(real_yield, breakeven, scenario, headlines)
        content_el.string = narrative
        set_state_comment(soup, "real_yield_state", current, content_el.parent)
    else:
        print("Real yield regime unchanged — keeping existing narrative.")


def update_etf_confluence_section(soup, gold_cot, gold_cvd, gold_quadrant,
                                    silver_cot, silver_cvd, silver_quadrant, headlines):
    content_el = soup.find(id="etf-confluence-content")
    if not content_el:
        print("WARNING: etf-confluence-content element not found — left unchanged.")
        return
    prior = get_prior_state(soup, "etf_confluence_state")
    current = {"gold": gold_quadrant, "silver": silver_quadrant}
    if prior != current or prior is None:
        print(f"ETF/COT confluence changed ({prior} -> {current}) — regenerating via AI.")
        gold_note = generate_etf_confluence_narrative("Gold", gold_cot, gold_cvd, gold_quadrant, headlines)
        silver_note = generate_etf_confluence_narrative("Silver", silver_cot, silver_cvd, silver_quadrant, headlines)
        content_el.clear()
        b_tag = soup.new_tag("b")
        b_tag.string = "Positioning Confluence (COT vs. ETF Flow):"
        content_el.append(b_tag)
        content_el.append(f" {gold_note} {silver_note}")
        set_state_comment(soup, "etf_confluence_state", current, content_el.parent)
    else:
        print("ETF/COT confluence unchanged — keeping existing narrative.")


def should_refresh_monthly(prior_state, current_trigger_flags):
    """Monthly narrative cadence, but responsive within the month: refresh if
    28+ days have passed since the last generation, OR if any of the passed
    trigger flags (e.g. risk regime, real yield scenario) changed."""
    if prior_state is None:
        return True
    try:
        last_gen = datetime.fromisoformat(prior_state.get("generated_at", ""))
        days_elapsed = (datetime.utcnow() - last_gen).days
    except Exception:
        days_elapsed = 999
    if days_elapsed >= 28:
        return True
    prior_flags = prior_state.get("flags", {})
    return prior_flags != current_trigger_flags


def generate_monetary_fiscal_narrative(fiscal_debt_trillion, real_yield, headlines):
    system_prompt = (
        "You write the 'Monetary & Fiscal Policy' section of a Q3 2026 institutional "
        "fundamental analysis panel, for gold and USD. Two short paragraphs: one framed "
        "'▲ USD' (near-term policy/rate factors), one '▼ USD' (structural fiscal factors). "
        "Use ONLY the DATA given — never invent a number not listed. Confident, terse, "
        "sell-side tone. Output plain HTML using only <b> and <br><br> for structure, "
        "matching this exact format: '<b>▲ USD</b> — [sentence]<br><br><b>▼ USD</b> — [sentence]'."
    )
    headline_sample = "; ".join(f'"{h}"' for h in (headlines or [])[:5]) or "none available"
    ry_line = f"10Y real yield: {real_yield['value']:.2f}% ({real_yield['delta']:+.2f})" if real_yield else "Real yield data unavailable"
    user_prompt = (
        f"US total public debt: ${fiscal_debt_trillion:.2f} trillion\n"
        f"{ry_line}\n"
        f"Headlines: {headline_sample}"
    )
    raw = call_ai_model(system_prompt, user_prompt, max_tokens=220)
    return raw.strip() if raw else None


def generate_geopolitical_narrative(headlines, risk_regime):
    system_prompt = (
        "You write the 'Geopolitical Risk & Sentiment' section of a Q3 2026 institutional "
        "fundamental analysis panel. Two short items: one '▲ USD', one '▼ USD', reflecting "
        "genuine tension in current conditions. Use ONLY the DATA given. Confident, terse tone. "
        "Output plain HTML: '<b>▲ USD</b> — [sentence]<br><br><b>▼ USD</b> — [sentence]'."
    )
    headline_sample = "; ".join(f'"{h}"' for h in (headlines or [])[:6]) or "none available"
    user_prompt = f"Risk regime: {risk_regime['status']}, Safe-Haven Flow: {risk_regime['safehaven_display']}\nHeadlines: {headline_sample}"
    raw = call_ai_model(system_prompt, user_prompt, max_tokens=200)
    return raw.strip() if raw else None


def generate_supply_demand_narrative(gold_cot, silver_cot, headlines):
    system_prompt = (
        "You write the 'Supply & Demand (Gold & Silver)' section of a Q3 2026 institutional "
        "fundamental analysis panel. Two short items, both framed '▼ USD' (structural demand "
        "factors), covering central bank reserve buying and physical market supply/demand. "
        "Use ONLY the DATA given — never invent a specific tonnage or deficit figure not listed. "
        "Output plain HTML: '<b>▼ USD</b> — [sentence]<br><br><b>▼ USD</b> — [sentence]'."
    )
    headline_sample = "; ".join(f'"{h}"' for h in (headlines or [])[:5]) or "none available"
    cot_line = ""
    if gold_cot: cot_line += f"Gold COT net: {gold_cot['net']:+,} ({gold_cot['delta']:+,} w/w). "
    if silver_cot: cot_line += f"Silver COT net: {silver_cot['net']:+,} ({silver_cot['delta']:+,} w/w)."
    user_prompt = f"{cot_line}\nHeadlines: {headline_sample}"
    raw = call_ai_model(system_prompt, user_prompt, max_tokens=200)
    return raw.strip() if raw else None


def generate_striking_note_narrative(headlines):
    system_prompt = (
        "You write a short institutional note (3-4 sentences) titled 'THE TARGET REVISION "
        "DECEPTION' — the point is to warn readers not to mistake sell-side banks trimming "
        "their price targets (on near-term rate repricing) for a reversal of the longer-run "
        "structural case. HARD RULE: do not state any specific new dollar price target or "
        "figure — this note is about how to interpret target revisions in general, not new "
        "numbers. Ground the framing in the real headlines given. Confident, terse tone, plain text."
    )
    headline_sample = "; ".join(f'"{h}"' for h in (headlines or [])[:5]) or "none available"
    user_prompt = f"Headlines: {headline_sample}"
    raw = call_ai_model(system_prompt, user_prompt, max_tokens=180)
    return raw.strip() if raw else None


def update_fundamental_narratives(soup, fiscal_debt_trillion, real_yield, risk_regime,
                                    gold_cot, silver_cot, headlines):
    """Materiality/monthly-gated update for the three original Fundamental Analysis
    subsections plus the Striking Institutional Note — all AI-generated (Gemini),
    grounded only in real fetched data, never inventing named-bank price figures."""
    trigger_flags = {
        "risk_status": risk_regime["status"] if risk_regime else None,
        "real_yield_delta_sign": (1 if real_yield and real_yield["delta"] > 0 else -1 if real_yield else 0),
    }
    prior = get_prior_state(soup, "fundamental_narrative_state")
    if not should_refresh_monthly(prior, trigger_flags):
        print("Fundamental Analysis narratives unchanged this run (within monthly window, no material trigger).")
        return

    print("Refreshing Fundamental Analysis narratives (monthly window elapsed or material trigger fired).")

    all_succeeded = True

    mf = generate_monetary_fiscal_narrative(fiscal_debt_trillion, real_yield, headlines)
    if mf:
        el = soup.find(id="monetary-fiscal-content")
        if el:
            el.clear()
            el.append(BeautifulSoup(mf, 'html.parser'))
    else:
        print("WARNING: Monetary & Fiscal narrative failed this run — left unchanged (stale).")
        all_succeeded = False
    time.sleep(3)

    geo = generate_geopolitical_narrative(headlines, risk_regime)
    if geo:
        el = soup.find(id="geopolitical-content")
        if el:
            el.clear()
            el.append(BeautifulSoup(geo, 'html.parser'))
    else:
        print("WARNING: Geopolitical narrative failed this run — left unchanged (stale).")
        all_succeeded = False
    time.sleep(3)

    sd = generate_supply_demand_narrative(gold_cot, silver_cot, headlines)
    if sd:
        el = soup.find(id="supply-demand-content")
        if el:
            el.clear()
            el.append(BeautifulSoup(sd, 'html.parser'))
    else:
        print("WARNING: Supply & Demand narrative failed this run — left unchanged (stale).")
        all_succeeded = False
    time.sleep(3)

    note = generate_striking_note_narrative(headlines)
    if note:
        el = soup.find(id="striking-note-content")
        if el:
            el.string = note
    else:
        print("WARNING: Striking Note narrative failed this run — left unchanged (stale).")
        all_succeeded = False

    if all_succeeded:
        anchor = soup.find(id="monetary-fiscal-section")
        set_state_comment(soup, "fundamental_narrative_state",
                           {"flags": trigger_flags, "generated_at": datetime.utcnow().isoformat()},
                           anchor)
    else:
        print("Not all Fundamental Analysis sections generated successfully — "
              "leaving the monthly-refresh timestamp unset so the next run retries "
              "the whole set, rather than treating this as done for the month.")


def fetch_treasury_debt_trillion():
    """Same public, no-key Treasury API already used client-side — Python
    equivalent so the monetary/fiscal narrative can ground on the real figure."""
    try:
        url = ("https://api.fiscaldata.treasury.gov/services/api/fiscal_service/"
               "v2/accounting/od/debt_to_penny?sort=-record_date&limit=1")
        d = fetch_json(url)
        row = d.get("data", [None])[0]
        if not row:
            return None
        return float(row["tot_pub_debt_out_amt"]) / 1e12
    except Exception as e:
        print(f"Treasury debt fetch failed: {e}")
        return None


def compute_risk_regime(vix_data, dxy_data, gold_data, credit_spread):
    """Transparent composite score — same principle as the bias badges:
    named thresholds, not a black box. 0 = max risk-off, 100 = max risk-on."""
    score = 50  # neutral baseline

    vix = vix_data['price'] if vix_data else None
    if vix is not None:
        if vix < 15: score += 20
        elif vix < 20: score += 5
        elif vix < 25: score -= 10
        else: score -= 25

    if credit_spread is not None:
        if credit_spread['delta'] < -0.02: score += 10   # narrowing = risk-on
        elif credit_spread['delta'] > 0.02: score -= 15   # widening = risk-off signal, weighted heavier

    safehaven_label = "Mixed"
    if dxy_data and gold_data:
        g_up = gold_data['pct_change'] > 0.2
        g_down = gold_data['pct_change'] < -0.2
        d_up = dxy_data['pct_change'] > 0.2
        d_down = dxy_data['pct_change'] < -0.2
        if g_up and d_down:
            safehaven_label = "Flight-to-Safety"; score -= 10
        elif g_down and d_up:
            safehaven_label = "Dollar-Preferred"; score += 10
        elif g_up and d_up:
            safehaven_label = "Mixed (Both Bid)"
        elif g_down and d_down:
            safehaven_label = "Mixed (Both Offered)"

    score = max(0, min(100, score))
    if score >= 70: status = "RISK-ON"
    elif score >= 55: status = "MODERATE RISK-ON"
    elif score >= 45: status = "NEUTRAL"
    elif score >= 30: status = "MODERATE RISK-OFF"
    else: status = "RISK-OFF"

    vix_display = f"~{vix:.1f}" if vix is not None else "N/A"
    if credit_spread is not None:
        credit_display = f"{credit_spread['value']:.2f}% ({'widening' if credit_spread['delta']>0 else 'narrowing' if credit_spread['delta']<0 else 'flat'})"
    else:
        credit_display = "Not Live (set FRED_API_KEY)"

    return {
        "status": status,
        "fill_pct": score,
        "vix_display": vix_display,
        "credit_display": credit_display,
        "safehaven_display": safehaven_label,
        "vix_raw": vix,
    }


def get_prior_risk_state(soup):
    """Reads the last run's status + VIX (embedded as an HTML comment) so we
    can decide whether today's move is material enough to regenerate the AI
    catalyst text, or whether to leave it alone — the 'weekly view, updates
    within the week if something happens' behavior."""
    comment = soup.find(string=lambda t: isinstance(t, Comment) and "risk_state:" in t)
    if not comment:
        return None
    try:
        payload = comment.split("risk_state:")[1].strip()
        return json.loads(payload)
    except Exception:
        return None


def is_material_change(prior, current):
    if prior is None:
        return True  # no prior state recorded — always generate the first time
    if prior.get("status") != current["status"]:
        return True
    prior_vix = prior.get("vix_raw")
    if prior_vix is not None and current["vix_raw"] is not None:
        if abs(current["vix_raw"] - prior_vix) >= 2.0:
            return True
    return False


def generate_catalyst_and_focus(headlines, risk_regime, gold_cot, silver_cot):
    """AI-generated only when is_material_change() says something actually
    shifted — otherwise the caller skips this entirely and keeps prior text."""
    headline_sample = "; ".join(f'"{h}"' for h in (headlines or [])[:5]) or "none available"
    system_prompt = (
        "You write two short institutional dashboard fields based ONLY on the DATA given:\n"
        "1. 'catalyst': the single most market-moving theme right now (under 12 words)\n"
        "2. 'affects': 2-4 asset classes it impacts, comma-separated, from this list only: "
        "Equities, Precious Metals, Energy, FX, Rates, Crypto\n"
        "3. 'focus': the dominant macro tension this quarter, framed as 'X vs Y' (under 8 words)\n"
        "Do not invent specific events not implied by the headlines. Output ONLY valid JSON: "
        '{"catalyst": "...", "affects": "...", "focus": "..."}'
    )
    user_prompt = (
        f"Headlines: {headline_sample}\n"
        f"Risk regime: {risk_regime['status']}, VIX {risk_regime['vix_display']}, "
        f"Safe-haven flow: {risk_regime['safehaven_display']}"
    )
    raw = call_ai_model(system_prompt, user_prompt, max_tokens=120)
    if raw:
        try:
            cleaned = raw.strip().strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
            parsed = json.loads(cleaned)
            if all(k in parsed for k in ("catalyst", "affects", "focus")):
                return parsed
        except Exception as e:
            print(f"Could not parse catalyst/focus AI response: {e}")
    return {
        "catalyst": "AI generation unavailable this run",
        "affects": "Equities, Precious Metals",
        "focus": "Sticky Inflation vs Deficits",
    }


def update_risk_gauge(soup, risk_regime, catalyst_data):
    box = soup.find("div", class_="risk-gauge-box")
    if box:
        status_el = box.find("div", class_="risk-status")
        if status_el: status_el.string = risk_regime["status"]
        fill_el = box.find("div", class_="risk-fill")
        if fill_el: fill_el["style"] = f"width:{risk_regime['fill_pct']}%"
        meta_el = box.find_all("div")[-1]
        if meta_el:
            meta_el.string = (f"VIX: {risk_regime['vix_display']} · "
                               f"Credit Spreads: {risk_regime['credit_display']} · "
                               f"Safe-Haven Flow: {risk_regime['safehaven_display']}")
    event_box = soup.find("div", class_="next-event-box")
    if event_box:
        title_el = event_box.find("div", style=lambda s: s and "font-weight:600" in s)
        if title_el: title_el.string = catalyst_data["catalyst"]
        affects_el = event_box.find("div", string=lambda t: t and "Affects:" in t)
        if affects_el: affects_el.string = f"Affects: {catalyst_data['affects']}"
        driver_el = event_box.find("div", string=lambda t: t and "Primary Driver:" in t)
        if driver_el: driver_el.string = f"Primary Driver: {catalyst_data['focus']}"
    # embed the state for next run's materiality check
    new_comment = Comment(f" risk_state:{json.dumps({'status': risk_regime['status'], 'vix_raw': risk_regime['vix_raw']})} ")
    old_comment = soup.find(string=lambda t: isinstance(t, Comment) and "risk_state:" in t)
    if old_comment:
        old_comment.replace_with(new_comment)
    elif box:
        box.insert_before(new_comment)
    print(f"Risk gauge updated: {risk_regime['status']} (fill {risk_regime['fill_pct']}%).")


def generate_updated_dashboard():
    with open('index.html', 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    session = determine_session()
    print(f"Running as: {session} session.")

    headlines = fetch_all_headlines()
    geo_summary = summarize_geopolitical_rate_context(headlines)

    # fetch gold/silver once, share between bias cards and the desk note paragraph
    gold_price = fetch_yahoo_chart("GC=F")
    silver_price = fetch_yahoo_chart("SI=F")
    gold_cot = fetch_cot_managed_money("GOLD - COMMODITY EXCHANGE")
    silver_cot = fetch_cot_managed_money("SILVER - COMMODITY EXCHANGE")

    bias_rows = compute_all_bias_rows(headlines, gold_price, silver_price, gold_cot, silver_cot)
    update_bias_cards(soup, bias_rows)

    liquidity_paragraph = generate_liquidity_paragraph(gold_price, silver_price, gold_cot, silver_cot, headlines, session, soup)
    update_desk_note(soup, geo_summary, liquidity_paragraph, session)
    time.sleep(3)  # spread AI call bursts across phases, not just within the bias-card loop

    # --- Risk Gauge: numbers computed every run, AI text only on material change ---
    vix_data = fetch_yahoo_chart("^VIX")
    dxy_data = fetch_yahoo_chart("DX-Y.NYB")
    credit_spread = fetch_fred_credit_spread()
    risk_regime = compute_risk_regime(vix_data, dxy_data, gold_price, credit_spread)

    prior_state = get_prior_risk_state(soup)
    if is_material_change(prior_state, risk_regime):
        print(f"Risk regime materially changed (prior: {prior_state}) — regenerating catalyst/focus via AI.")
        catalyst_data = generate_catalyst_and_focus(headlines, risk_regime, gold_cot, silver_cot)
        update_risk_gauge(soup, risk_regime, catalyst_data)
    else:
        print("No material change in risk regime — updating live numbers only, keeping existing catalyst/focus text.")
        box = soup.find("div", class_="risk-gauge-box")
        if box:
            status_el = box.find("div", class_="risk-status")
            if status_el: status_el.string = risk_regime["status"]
            fill_el = box.find("div", class_="risk-fill")
            if fill_el: fill_el["style"] = f"width:{risk_regime['fill_pct']}%"
            meta_el = box.find_all("div")[-1]
            if meta_el:
                meta_el.string = (f"VIX: {risk_regime['vix_display']} · "
                                   f"Credit Spreads: {risk_regime['credit_display']} · "
                                   f"Safe-Haven Flow: {risk_regime['safehaven_display']}")
        new_comment = Comment(f" risk_state:{json.dumps({'status': risk_regime['status'], 'vix_raw': risk_regime['vix_raw']})} ")
        old_comment = soup.find(string=lambda t: isinstance(t, Comment) and "risk_state:" in t)
        if old_comment:
            old_comment.replace_with(new_comment)
        elif box:
            box.insert_before(new_comment)

    time.sleep(3)

    # --- Real Yields & Rate Transmission (monthly narrative, materiality-gated) ---
    fred_key = os.environ.get("FRED_API_KEY")
    real_yield = fetch_fred_series("DFII10", fred_key)
    breakeven = fetch_fred_series("T10YIE", fred_key)
    ry_scenario = classify_real_yield_regime(real_yield, breakeven)
    update_real_yield_section(soup, real_yield, breakeven, ry_scenario, headlines)
    time.sleep(3)

    # --- ETF-flow vs COT confluence (folded into Supply & Demand, materiality-gated) ---
    gld_data = fetch_yahoo_chart("GLD")
    slv_data = fetch_yahoo_chart("SLV")
    gold_cvd = compute_volume_delta_zone(gld_data) if gld_data else None
    silver_cvd = compute_volume_delta_zone(slv_data) if slv_data else None
    gold_quadrant = classify_etf_cot_confluence(gold_cot, gold_cvd)
    silver_quadrant = classify_etf_cot_confluence(silver_cot, silver_cvd)
    update_etf_confluence_section(soup, gold_cot, gold_cvd, gold_quadrant,
                                    silver_cot, silver_cvd, silver_quadrant, headlines)
    time.sleep(3)

    # --- Fundamental Analysis (Monetary/Fiscal, Geopolitical, Supply&Demand) + Striking Note ---
    # Monthly narrative, but responsive within the month if risk regime or real yield scenario shifts
    fiscal_debt = fetch_treasury_debt_trillion()
    update_fundamental_narratives(soup, fiscal_debt, real_yield, risk_regime,
                                    gold_cot, silver_cot, headlines)

    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(str(soup))

    print("index.html updated successfully.")


if __name__ == '__main__':
    generate_updated_dashboard()
