import os
import shutil
import base64
import re
import itertools
from fastapi import FastAPI, UploadFile, File, Form
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

print("DEBUG: SERVER STARTING...", flush=True)

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

def call_vision_model(msg):
    """Smart Search: Tries multiple model names automatically."""
    
    # 1. Try Google Gemini with 3 different possible names
    # Your logs showed 'gemini-1.5-flash-latest' failed, so we try the standard ones.
    gemini_names = ["gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-pro-vision"]
    
    if google_keys:
        for _ in range(len(google_keys)):
            current_key = next(key_iterator)
            for model_name in gemini_names:
                try:
                    print(f"DEBUG: Trying Google Key with {model_name}...", flush=True)
                    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, api_key=current_key)
                    response = llm.invoke([msg])
                    print(f"DEBUG: SUCCESS with {model_name}", flush=True)
                    return clean_ai_text(response.content)
                except Exception as e:
                    print(f"DEBUG: {model_name} failed: {str(e)}", flush=True)
                    continue

    # 2. Try OpenRouter (Using the most stable Free Vision slug)
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        # 'google/gemini-2.0-flash-exp:free' is the current correct free slug
        try:
            print("DEBUG: Trying OpenRouter (gemini-2.0-flash-exp:free)...", flush=True)
            llm = ChatOpenAI(
                model="google/gemini-2.0-flash-exp:free", 
                api_key=openrouter_key, 
                base_url="https://openrouter.ai/api/v1"
            )
            response = llm.invoke([msg])
            print("DEBUG: OpenRouter SUCCESS", flush=True)
            return clean_ai_text(response.content)
        except Exception as e:
            print(f"DEBUG: OpenRouter FAILED: {str(e)}", flush=True)

    # 3. Try Groq (Trying the 90B model since your logs said 11B was missing)
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        groq_names = ["llama-3.2-90b-vision-preview", "llama-3.2-11b-vision-instruct"]
        for g_model in groq_names:
            try:
                print(f"DEBUG: Trying Groq with {g_model}...", flush=True)
                llm = ChatGroq(model=g_model, temperature=0, api_key=groq_key)
                response = llm.invoke([msg])
                print(f"DEBUG: Groq SUCCESS with {g_model}", flush=True)
                return clean_ai_text(response.content)
            except Exception as e:
                print(f"DEBUG: Groq {g_model} FAILED: {str(e)}", flush=True)
                continue

    return "Network error. Please contact Zubair for support."

@app.post("/analyze-scene")
async def analyze_scene(image: UploadFile = File(...)):
    temp_path = f"temp_norm_{image.filename}"
    try:
        with open(temp_path, "wb") as buffer: shutil.copyfileobj(image.file, buffer)
        with open(temp_path, "rb") as img_file: base64_image = base64.b64encode(img_file.read()).decode('utf-8')
        prompt = "Describe the scene for a blind person. Be conversational and warn of hazards."
        msg = HumanMessage(content=[{"type": "text", "text": prompt},{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}])
        result = call_vision_model(msg)
        if os.path.exists(temp_path): os.remove(temp_path)
        return {"status": "success", "script": result}
    except Exception:
        return {"status": "error", "message": "Connection lost. Please contact Zubair for support."}

@app.post("/ask-vision")
async def ask_vision(image: UploadFile = File(...), question: str = Form(...)):
    temp_path = f"temp_ask_{image.filename}"
    try:
        with open(temp_path, "wb") as buffer: shutil.copyfileobj(image.file, buffer)
        with open(temp_path, "rb") as img_file: base64_image = base64.b64encode(img_file.read()).decode('utf-8')
        prompt = f"The user asks: {question}. Tell them where the object is or guide them."
        msg = HumanMessage(content=[{"type": "text", "text": prompt},{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}])
        result = call_vision_model(msg)
        if os.path.exists(temp_path): os.remove(temp_path)
        return {"status": "success", "script": result}
    except Exception:
        return {"status": "error", "message": "Connection lost. Please contact Zubair for support."}

@app.post("/navigate")
async def navigate(image: UploadFile = File(...)):
    temp_path = f"temp_nav_{image.filename}"
    try:
        with open(temp_path, "wb") as buffer: shutil.copyfileobj(image.file, buffer)
        with open(temp_path, "rb") as img_file: base64_image = base64.b64encode(img_file.read()).decode('utf-8')
        prompt = "Walking guide: 1 short sentence about what is directly ahead and distance."
        msg = HumanMessage(content=[{"type": "text", "text": prompt},{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}])
        result = call_vision_model(msg)
        if os.path.exists(temp_path): os.remove(temp_path)
        return {"status": "success", "script": result}
    except Exception:
        return {"status": "error", "message": "Connection lost. Please contact Zubair for support."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
