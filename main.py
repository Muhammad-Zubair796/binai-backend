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
        genai.configure(api_key=google_keys[0])
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                # This will print the EXACT names Google wants
                print(f"DEBUG: WORKING GOOGLE MODEL FOUND: {m.name}", flush=True)
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
    
    # 1. Try Google Gemini (Trying the 3 most likely names)
    gemini_names = ["gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-1.5-pro"]
    if google_keys:
        for _ in range(len(google_keys)):
            current_key = next(key_iterator)
            genai.configure(api_key=current_key)
            for g_name in gemini_names:
                try:
                    print(f"DEBUG: Trying Google Key with {g_name}...", flush=True)
                    model = genai.GenerativeModel(g_name)
                    img = Image.open(io.BytesIO(image_bytes))
                    response = model.generate_content([prompt, img])
                    print(f"DEBUG: SUCCESS with {g_name}", flush=True)
                    return clean_ai_text(response.text)
                except Exception as e:
                    print(f"DEBUG: Google {g_name} failed: {str(e)}", flush=True)
                    continue

    # 2. Try OpenRouter (Using the current free vision slug)
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        # This is the most common free vision slug on OpenRouter
        or_models = ["google/gemini-2.0-flash-exp:free", "google/gemini-flash-1.5-exp:free"]
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

    # 3. Try Groq (Trying both 11B and 90B versions)
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        groq_models = ["llama-3.2-11b-vision-instruct", "llama-3.2-90b-vision-instruct"]
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
