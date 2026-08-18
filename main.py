import os
import shutil
import base64
import re
import itertools
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, Form
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from PIL import Image
import io

print("DEBUG: SERVER STARTING - BULLETPROOF EDITION", flush=True)

app = FastAPI(title="binAI Human Assistant Backend")

# ==========================================
# API KEY ROTATOR
# ==========================================
google_keys_env = os.getenv("GOOGLE_API_KEYS", "")
google_keys = [k.strip() for k in google_keys_env.split(",") if k.strip()]
key_iterator = itertools.cycle(google_keys) if google_keys else None

@app.get("/")
@app.head("/")
async def health_check():
    return {"status": "alive", "message": "binAI Human Assistant is running!"}

def clean_ai_text(raw_text):
    clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL | re.IGNORECASE)
    clean_text = re.sub(r'<thought>.*?</thought>', '', clean_text, flags=re.DOTALL | re.IGNORECASE)
    return clean_text.replace('<', '').replace('>', '').strip()

def call_vision_model(prompt, image_bytes):
    """Bulletproof Router: Tries multiple models per provider automatically."""
    
    # Get models from Environment Variables (with safe, currently active defaults)
    env_g_model = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
    env_or_model = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-lite-preview-02-05:free")
    env_groq_model = os.getenv("GROQ_MODEL", "llama-3.2-90b-vision-preview") # 11b was decommissioned!

    # 1. Try Google Gemini (Tries Env Var, then 2.0, then 1.5)
    if google_keys:
        google_models_to_try = [env_g_model, 'gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-pro']
        
        for i in range(len(google_keys)):
            current_key = next(key_iterator)
            genai.configure(api_key=current_key)
            
            for g_model in google_models_to_try:
                try:
                    print(f"DEBUG: Trying Google Key #{i+1} with {g_model}...", flush=True)
                    model = genai.GenerativeModel(g_model)
                    img = Image.open(io.BytesIO(image_bytes))
                    response = model.generate_content([prompt, img])
                    print(f"DEBUG: Google SUCCESS with {g_model}", flush=True)
                    return clean_ai_text(response.text)
                except Exception as e:
                    print(f"DEBUG: Google {g_model} FAILED: {str(e)}", flush=True)
                    continue # Try the next Google model in the list

    # 2. Try OpenRouter (Tries Env Var, then Gemini Lite Free, then Qwen Free)
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        or_models_to_try = [env_or_model, "google/gemini-2.0-flash-lite-preview-02-05:free", "qwen/qwen-vl-plus:free"]
        
        for or_model in or_models_to_try:
            try:
                print(f"DEBUG: Trying OpenRouter with {or_model}...", flush=True)
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                llm = ChatOpenAI(model=or_model, api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")
                msg = HumanMessage(content=[{"type": "text", "text": prompt},{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}])
                response = llm.invoke([msg])
                print(f"DEBUG: OpenRouter SUCCESS with {or_model}", flush=True)
                return clean_ai_text(response.content)
            except Exception as e:
                print(f"DEBUG: OpenRouter {or_model} FAILED: {str(e)}", flush=True)
                continue # Try the next OpenRouter model in the list

    # 3. Try Groq (Using the 90b model since 11b is dead)
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            print(f"DEBUG: Trying Groq with {env_groq_model}...", flush=True)
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            llm = ChatGroq(model=env_groq_model, temperature=0, api_key=groq_key)
            msg = HumanMessage(content=[{"type": "text", "text": prompt},{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}])
            response = llm.invoke([msg])
            print(f"DEBUG: Groq SUCCESS", flush=True)
            return clean_ai_text(response.content)
        except Exception as e:
            print(f"DEBUG: Groq FAILED: {str(e)}", flush=True)

    return "Network error. Please contact Zubair for support."

@app.post("/analyze-scene")
async def analyze_scene(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        prompt = "Describe the scene for a blind person. Be conversational and warn of hazards."
        result = call_vision_model(prompt, image_bytes)
        return {"status": "success", "script": result}
    except Exception:
        return {"status": "error", "message": "Connection lost. Please contact Zubair for support."}

@app.post("/ask-vision")
async def ask_vision(image: UploadFile = File(...), question: str = Form(...)):
    try:
        image_bytes = await image.read()
        prompt = f"The user asks: {question}. Tell them where the object is or guide them."
        result = call_vision_model(prompt, image_bytes)
        return {"status": "success", "script": result}
    except Exception:
        return {"status": "error", "message": "Connection lost. Please contact Zubair for support."}

@app.post("/navigate")
async def navigate(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        prompt = "Walking guide: 1 short sentence about what is directly ahead and distance."
        result = call_vision_model(prompt, image_bytes)
        return {"status": "success", "script": result}
    except Exception:
        return {"status": "error", "message": "Connection lost. Please contact Zubair for support."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
