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

app = FastAPI(title="binAI Human Assistant Backend")

# ==========================================
# API KEY ROTATOR
# ==========================================
google_keys_env = os.getenv("GOOGLE_API_KEYS", "")
google_keys = [k.strip() for k in google_keys_env.split(",") if k.strip()]
key_iterator = itertools.cycle(google_keys) if google_keys else None

print(f"DEBUG: Loaded {len(google_keys)} Google Keys.")
print(f"DEBUG: SambaNova Key Present: {bool(os.getenv('SAMBANOVA_API_KEY'))}")
print(f"DEBUG: Groq Key Present: {bool(os.getenv('GROQ_API_KEY'))}")

@app.get("/")
@app.head("/")
async def health_check():
    return {"status": "alive", "message": "binAI Human Assistant is running!"}

def clean_ai_text(raw_text):
    clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL | re.IGNORECASE)
    clean_text = re.sub(r'<thought>.*?</thought>', '', clean_text, flags=re.DOTALL | re.IGNORECASE)
    return clean_text.replace('<', '').replace('>', '').strip()

def call_vision_model(msg, fast_mode=False):
    """Tries 4 Google keys -> SambaNova -> Groq."""
    
    # 1. Try Google Gemini
    if google_keys:
        for i in range(len(google_keys)):
            current_key = next(key_iterator)
            try:
                print(f"DEBUG: Attempting Google Key #{i+1}")
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, api_key=current_key)
                response = llm.invoke([msg])
                print("DEBUG: Google Success!")
                return clean_ai_text(response.content)
            except Exception as e:
                print(f"DEBUG: Google Key #{i+1} Failed: {str(e)}")
                continue

    # 2. Try SambaNova
    sambanova_key = os.getenv("SAMBANOVA_API_KEY")
    if sambanova_key:
        try:
            print("DEBUG: Attempting SambaNova (Llama 3.2 Vision)")
            llm = ChatOpenAI(model="Llama-3.2-11B-Vision-Instruct", api_key=sambanova_key, base_url="https://api.sambanova.ai/v1")
            response = llm.invoke([msg])
            print("DEBUG: SambaNova Success!")
            return clean_ai_text(response.content)
        except Exception as e:
            print(f"DEBUG: SambaNova Failed: {str(e)}")

    # 3. Try Groq
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            print("DEBUG: Attempting Groq (Llama 3.2 Vision)")
            llm = ChatGroq(model="llama-3.2-11b-vision-instruct", temperature=0, api_key=groq_key)
            response = llm.invoke([msg])
            print("DEBUG: Groq Success!")
            return clean_ai_text(response.content)
        except Exception as e:
            print(f"DEBUG: Groq Failed: {str(e)}")

    return "Network error. Please contact Zubair for support."

@app.post("/analyze-scene")
async def analyze_scene(image: UploadFile = File(...)):
    print("DEBUG: Received /analyze-scene request")
    temp_path = f"temp_norm_{image.filename}"
    try:
        with open(temp_path, "wb") as buffer: shutil.copyfileobj(image.file, buffer)
        with open(temp_path, "rb") as img_file: base64_image = base64.b64encode(img_file.read()).decode('utf-8')
        prompt = "Describe the scene for a blind person. Be conversational and warn of hazards."
        msg = HumanMessage(content=[{"type": "text", "text": prompt},{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}])
        result = call_vision_model(msg)
        if os.path.exists(temp_path): os.remove(temp_path)
        return {"status": "success", "script": result}
    except Exception as e:
        print(f"DEBUG: Critical Error in analyze-scene: {str(e)}")
        return {"status": "error", "message": "Connection lost. Please contact Zubair for support."}

@app.post("/ask-vision")
async def ask_vision(image: UploadFile = File(...), question: str = Form(...)):
    print(f"DEBUG: Received /ask-vision request: {question}")
    temp_path = f"temp_ask_{image.filename}"
    try:
        with open(temp_path, "wb") as buffer: shutil.copyfileobj(image.file, buffer)
        with open(temp_path, "rb") as img_file: base64_image = base64.b64encode(img_file.read()).decode('utf-8')
        prompt = f"The user asks: {question}. Tell them where the object is or guide them."
        msg = HumanMessage(content=[{"type": "text", "text": prompt},{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}])
        result = call_vision_model(msg)
        if os.path.exists(temp_path): os.remove(temp_path)
        return {"status": "success", "script": result}
    except Exception as e:
        print(f"DEBUG: Critical Error in ask-vision: {str(e)}")
        return {"status": "error", "message": "Connection lost. Please contact Zubair for support."}

@app.post("/navigate")
async def navigate(image: UploadFile = File(...)):
    print("DEBUG: Received /navigate request")
    temp_path = f"temp_nav_{image.filename}"
    try:
        with open(temp_path, "wb") as buffer: shutil.copyfileobj(image.file, buffer)
        with open(temp_path, "rb") as img_file: base64_image = base64.b64encode(img_file.read()).decode('utf-8')
        prompt = "Walking guide: 1 short sentence about what is directly ahead and distance."
        msg = HumanMessage(content=[{"type": "text", "text": prompt},{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}])
        result = call_vision_model(msg, fast_mode=True)
        if os.path.exists(temp_path): os.remove(temp_path)
        return {"status": "success", "script": result}
    except Exception as e:
        print(f"DEBUG: Critical Error in navigate: {str(e)}")
        return {"status": "error", "message": "Connection lost. Please contact Zubair for support."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
