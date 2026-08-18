import os
import shutil
import base64
import re
import itertools
from google import genai
from google.genai import types
from fastapi import FastAPI, UploadFile, File, Form
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from PIL import Image
import io

print("DEBUG: SERVER STARTING...", flush=True)

app = FastAPI(title="binAI Human Assistant Backend")

# ==========================================
# API KEY ROTATOR
# ==========================================
google_keys_env = os.getenv("GOOGLE_API_KEYS", "")
google_keys = [k.strip() for k in google_keys_env.split(",") if k.strip()]
key_iterator = itertools.cycle(google_keys) if google_keys else None

# YOUR IDEA: List all working models for your keys
if google_keys:
    try:
        print(f"DEBUG: Scanning available models for your Google Key...", flush=True)
        scan_client = genai.Client(api_key=google_keys[0])
        for m in scan_client.models.list():
            print(f"DEBUG: GOOGLE MODEL FOUND: {m.name}", flush=True)
    except Exception as e:
        print(f"DEBUG: Google Model Scan Failed: {str(e)}", flush=True)

@app.get("/")
@app.head("/")
async def health_check():
    return {"status": "alive", "message": "binAI Human Assistant is running!"}

def clean_ai_text(raw_text):
    clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL | re.IGNORECASE)
    clean_text = re.sub(r'<thought>.*?</thought>', '', clean_text, flags=re.DOTALL | re.IGNORECASE)
    return clean_text.replace('<', '').replace('>', '').strip()

def call_vision_model(prompt, image_bytes):
    """Tries multiple names for Google -> OpenRouter -> Groq."""
    
    # 1. Try Google Gemini
    if google_keys:
        for _ in range(len(google_keys)):
            current_key = next(key_iterator)
            try:
                print(f"DEBUG: Trying Google Key with gemini-2.5-flash-lite...", flush=True)
                client = genai.Client(api_key=current_key)

                response = client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=[
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type="image/jpeg"
                        ),
                        prompt
                    ]
                )

                print(f"DEBUG: SUCCESS with gemini-2.5-flash-lite", flush=True)
                return clean_ai_text(response.text)
            except Exception as e:
                print(f"DEBUG: Google gemini-2.5-flash-lite failed: {str(e)}", flush=True)
                continue

    # 2. Try OpenRouter (Using the current free router)
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        or_models = ["openrouter/free"]
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        for or_m in or_models:
            try:
                print(f"DEBUG: Trying OpenRouter with {or_m}...", flush=True)
                llm = ChatOpenAI(model=or_m, api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")
                msg = HumanMessage(content=[{"type": "text", "text": prompt},{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}])
                response = llm.invoke([msg])
                print(f"DEBUG: OpenRouter SUCCESS with {or_m}", flush=True)
                return clean_ai_text(response.content)
            except Exception as e:
                print(f"DEBUG: OpenRouter {or_m} FAILED: {str(e)}", flush=True)

    # 3. Try Groq
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        groq_models = ["qwen/qwen3.6-27b"]
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        for gr_m in groq_models:
            try:
                print(f"DEBUG: Trying Groq with {gr_m}...", flush=True)
                llm = ChatGroq(model=gr_m, temperature=0, api_key=groq_key)
                msg = HumanMessage(content=[{"type": "text", "text": prompt},{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}])
                response = llm.invoke([msg])
                print(f"DEBUG: Groq SUCCESS with {gr_m}", flush=True)
                return clean_ai_text(response.content)
            except Exception as e:
                print(f"DEBUG: Groq {gr_m} FAILED: {str(e)}", flush=True)

    return "Network error. Please contact Zubair for support."

@app.post("/analyze-scene")
async def analyze_scene(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        prompt = "Describe the scene for a blind person. Be conversational and warn of hazards."
        result = call_vision_model(prompt, image_bytes)
        return {"status": "success", "script": result}
    except Exception as e:
        print(f"DEBUG: Analyze Scene Error: {str(e)}", flush=True)
        return {"status": "error", "message": "Connection lost. Please contact Zubair for support."}

@app.post("/ask-vision")
async def ask_vision(image: UploadFile = File(...), question: str = Form(...)):
    try:
        image_bytes = await image.read()
        prompt = f"The user asks: {question}. Tell them where the object is or guide them."
        result = call_vision_model(prompt, image_bytes)
        return {"status": "success", "script": result}
    except Exception as e:
        print(f"DEBUG: Ask Vision Error: {str(e)}", flush=True)
        return {"status": "error", "message": "Connection lost. Please contact Zubair for support."}

@app.post("/navigate")
async def navigate(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        prompt = "Walking guide: 1 short sentence about what is directly ahead and distance."
        result = call_vision_model(prompt, image_bytes)
        return {"status": "success", "script": result}
    except Exception as e:
        print(f"DEBUG: Navigate Error: {str(e)}", flush=True)
        return {"status": "error", "message": "Connection lost. Please contact Zubair for support."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
