import os
import urllib.request
import json

def generate_ai_desk_note(macro_headlines, te_calendar_event):
    """
    Sends live feeds to an LLM API (e.g., OpenAI) with a strict institutional 
    prompt to generate the session desk note dynamically.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return (
            "<b>GEOPOLITICAL &amp; RATE TRANSMISSION:</b> Fallback mode — OPENAI_API_KEY not found. "
            f"Latest market pulse: {macro_headlines[0]}<br><br>"
            "<b>STRUCTURAL PHYSICAL DEFICIT:</b> Sustained industrial demand and reserve accumulation remain active."
        )

    prompt = f"""
    You are an institutional trading desk macro strategist. 
    Analyze the following live inputs:
    - Latest Macro Headine 1: {macro_headlines[0]}
    - Latest Macro Headine 2: {macro_headlines[1]}
    - Upcoming High-Impact Calendar Event: {te_calendar_event or 'None'}

    Write the "INSTITUTIONAL SESSION DESK NOTE — MACRO REGIME" section consisting of two distinct, bolded paragraphs:
    1. GEOPOLITICAL & RATE TRANSMISSION: Focus on central bank rate expectations, real yields, and geopolitical risk factors.
    2. STRUCTURAL PHYSICAL DEFICIT: Focus on structural supply shortages, physical commodities, or tech/industrial capital flows.
    
    Keep the tone sharp, professional, concise, and institutional. Format output strictly as two HTML paragraphs with appropriate bold headers.
    """

    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode('utf-8'),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            return res_data['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"AI Generation failed: {e}")
        return "<b>GEOPOLITICAL &amp; RATE TRANSMISSION:</b> API generation error. Desk note running on default parameters.<br><br><b>STRUCTURAL PHYSICAL DEFICIT:</b> Core metrics unchanged."
