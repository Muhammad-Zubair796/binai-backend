import os
import shutil
import base64
import re
from fastapi import FastAPI, UploadFile, File, Form
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

app = FastAPI(title="binAI Human Assistant Backend")

@app.get("/")
async def health_check():
    return {"status": "alive", "message": "binAI Human Assistant is running!"}

def clean_ai_text(raw_text):
    """Removes <think> tags and stray symbols so TTS reads it perfectly."""
    clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL | re.IGNORECASE)
    clean_text = re.sub(r'<thought>.*?</thought>', '', clean_text, flags=re.DOTALL | re.IGNORECASE)
    return clean_text.replace('<', '').replace('>', '').strip()

def call_groq_vision(msg, fast_mode=False):
    """Smart fallback loop to try AI models."""
    if fast_mode:
        models = ["llama-3.2-11b-vision-instruct", "qwen/qwen3.6-27b"]
    else:
        models = ["llama-3.2-11b-vision-instruct", "llama-3.2-90b-vision-instruct", "qwen/qwen3.6-27b"]
        
    last_error = "Unknown error"
    for model_name in models:
        try:
            llm = ChatGroq(model=model_name, temperature=0, api_key=os.getenv("GROQ_API_KEY"))
            response = llm.invoke([msg])
            return clean_ai_text(response.content)
        except Exception as e:
            last_error = str(e)
            continue
    return None

# ==========================================
# MODE 1: NORMAL DETECTION (Triggered by Double Tap)
# ==========================================
@app.post("/analyze-scene")
async def analyze_scene(image: UploadFile = File(...)):
    temp_path = f"temp_norm_{image.filename}"
    try:
        with open(temp_path, "wb") as buffer: shutil.copyfileobj(image.file, buffer)
        with open(temp_path, "rb") as img_file: base64_image = base64.b64encode(img_file.read()).decode('utf-8')

        prompt = """You are a human-like assistant for a blind person. Describe the scene in front of them.
        RULES:
        1. Be friendly and conversational.
        2. Warn about hazards immediately.
        3. Output ONLY the spoken words."""

        msg = HumanMessage(content=[{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}])
        result = call_groq_vision(msg)
        
        if os.path.exists(temp_path): os.remove(temp_path)
        return {"status": "success", "script": result} if result else {"status": "error", "message": "AI failed."}
    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        return {"status": "error", "message": str(e)}

# ==========================================
# MODE 2: SEARCH & ASSIST (Triggered by Swipe Up + Voice)
# ==========================================
@app.post("/ask-vision")
async def ask_vision(image: UploadFile = File(...), question: str = Form(...)):
    temp_path = f"temp_ask_{image.filename}"
    try:
        with open(temp_path, "wb") as buffer: shutil.copyfileobj(image.file, buffer)
        with open(temp_path, "rb") as img_file: base64_image = base64.b64encode(img_file.read()).decode('utf-8')

        prompt = f"""You are a human assistant helping a blind person find something. They asked: "{question}"
        RULES:
        1. IF YOU SEE IT: Tell them exactly where it is (e.g., 'Your medicine is on the table to your right').
        2. IF YOU DO NOT SEE IT: Act like a human guiding them. Say you don't see it, and tell them to move the camera or look somewhere else (e.g., 'I don't see it here. Try moving your camera to the left, or open the drawer in front of you').
        3. Output ONLY the spoken words."""

        msg = HumanMessage(content=[{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}])
        result = call_groq_vision(msg)
        
        if os.path.exists(temp_path): os.remove(temp_path)
        return {"status": "success", "script": result} if result else {"status": "error", "message": "AI failed."}
    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        return {"status": "error", "message": str(e)}

# ==========================================
# MODE 3: WALK MODE (Triggered by Long Press)
# ==========================================
@app.post("/navigate")
async def navigate(image: UploadFile = File(...)):
    temp_path = f"temp_nav_{image.filename}"
    try:
        with open(temp_path, "wb") as buffer: shutil.copyfileobj(image.file, buffer)
        with open(temp_path, "rb") as img_file: base64_image = base64.b64encode(img_file.read()).decode('utf-8')

        prompt = """You are a real-time walking guide for a blind person. Safety is your priority.
        RULES:
        1. Keep it extremely short (1 sentence).
        2. Estimate distance and warn of hazards (e.g., 'Door open 1 meter ahead', 'Stairs ahead, step carefully').
        3. If blurry, say: 'Image is not clear, step carefully'.
        4. Output ONLY the spoken words."""

        msg = HumanMessage(content=[{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}])
        result = call_groq_vision(msg, fast_mode=True) 
        
        if os.path.exists(temp_path): os.remove(temp_path)
        return {"status": "success", "script": result} if result else {"status": "error", "message": "AI failed."}
    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
