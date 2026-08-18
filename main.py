import os
import shutil
import base64
import re
import itertools
from fastapi import FastAPI, UploadFile, File, Form
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

app = FastAPI(title="binAI Human Assistant Backend")

# ==========================================
# API KEY ROTATOR SETUP
# ==========================================
google_keys_env = os.getenv("GOOGLE_API_KEYS", os.getenv("GOOGLE_API_KEY", ""))
google_keys = [k.strip() for k in google_keys_env.split(",") if k.strip()]
key_iterator = itertools.cycle(google_keys) if google_keys else None

@app.get("/")
@app.head("/")
async def health_check():
    print("Health check endpoint pinged.")
    return {"status": "alive", "message": "binAI Human Assistant is running!"}

def clean_ai_text(raw_text):
    """Removes <think> tags and stray symbols so TTS reads it perfectly."""
    clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL | re.IGNORECASE)
    clean_text = re.sub(r'<thought>.*?</thought>', '', clean_text, flags=re.DOTALL | re.IGNORECASE)
    return clean_text.replace('<', '').replace('>', '').strip()

def call_vision_model(msg, fast_mode=False):
    """Smart fallback loop: Tries ALL Google Gemini keys, then falls back to Groq."""
    last_error = "Unknown error"
    
    # 1. Try Google Gemini Keys (Loop through all available keys)
    if google_keys:
        for _ in range(len(google_keys)):
            current_key = next(key_iterator)
            try:
                print(f"Attempting Google Gemini with rotated key...")
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, api_key=current_key)
                response = llm.invoke([msg])
                print("Success! Google Gemini generated a response.")
                return clean_ai_text(response.content)
            except Exception as e:
                last_error = str(e)
                print(f"Key failed, trying next... Error: {last_error}")
                continue

    # 2. Fallback to Groq if all Gemini keys fail
    groq_models = ["llama-3.2-11b-vision-instruct"] if fast_mode else ["llama-3.2-11b-vision-instruct", "llama-3.2-90b-vision-instruct"]
    
    for model_name in groq_models:
        try:
            print(f"Attempting Groq model: {model_name}")
            llm = ChatGroq(model=model_name, temperature=0, api_key=os.getenv("GROQ_API_KEY"))
            response = llm.invoke([msg])
            print(f"Success! Groq model {model_name} generated a response.")
            return clean_ai_text(response.content)
        except Exception as e:
            last_error = str(e)
            print(f"FAILED Groq model {model_name}. Error: {last_error}")
            continue

    print(f"CRITICAL: ALL models failed. Last error: {last_error}")
    return None

@app.post("/analyze-scene")
async def analyze_scene(image: UploadFile = File(...)):
    temp_path = f"temp_norm_{image.filename}"
    try:
        with open(temp_path, "wb") as buffer: shutil.copyfileobj(image.file, buffer)
        with open(temp_path, "rb") as img_file: base64_image = base64.b64encode(img_file.read()).decode('utf-8')
        prompt = """You are a human-like assistant for a blind person. Describe the scene in front of them.
        RULES: 1. Be friendly and conversational. 2. Warn about hazards immediately. 3. Output ONLY the spoken words."""
        msg = HumanMessage(content=[{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}])
        result = call_vision_model(msg)
        if os.path.exists(temp_path): os.remove(temp_path)
        return {"status": "success", "script": result} if result else {"status": "error", "message": "AI failed."}
    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        return {"status": "error", "message": str(e)}

@app.post("/ask-vision")
async def ask_vision(image: UploadFile = File(...), question: str = Form(...)):
    temp_path = f"temp_ask_{image.filename}"
    try:
        with open(temp_path, "wb") as buffer: shutil.copyfileobj(image.file, buffer)
        with open(temp_path, "rb") as img_file: base64_image = base64.b64encode(img_file.read()).decode('utf-8')
        prompt = f"""You are a human assistant helping a blind person find something. They asked: "{question}"
        RULES: 1. IF YOU SEE IT: Tell them exactly where it is. 2. IF YOU DO NOT SEE IT: Act like a human guiding them. 3. Output ONLY the spoken words."""
        msg = HumanMessage(content=[{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}])
        result = call_vision_model(msg)
        if os.path.exists(temp_path): os.remove(temp_path)
        return {"status": "success", "script": result} if result else {"status": "error", "message": "AI failed."}
    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        return {"status": "error", "message": str(e)}

@app.post("/navigate")
async def navigate(image: UploadFile = File(...)):
    temp_path = f"temp_nav_{image.filename}"
    try:
        with open(temp_path, "wb") as buffer: shutil.copyfileobj(image.file, buffer)
        with open(temp_path, "rb") as img_file: base64_image = base64.b64encode(img_file.read()).decode('utf-8')
        prompt = """You are a real-time walking guide for a blind person. Safety is your priority.
        RULES: 1. Keep it extremely short (1 sentence). 2. Estimate distance and warn of hazards. 3. Output ONLY the spoken words."""
        msg = HumanMessage(content=[{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}])
        result = call_vision_model(msg, fast_mode=True) 
        if os.path.exists(temp_path): os.remove(temp_path)
        return {"status": "success", "script": result} if result else {"status": "error", "message": "AI failed."}
    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
