import os
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

print("DEBUG: SERVER STARTING - BULLETPROOF EDITION V4.2 (2026 MODELS + OPTIMIZED)", flush=True)

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
    """Bulletproof Router: Tries multiple models per provider automatically with timeouts."""
    
    env_g_model = os.getenv("GOOGLE_MODEL", "")
    env_or_model = os.getenv("OPENROUTER_MODEL", "")
    env_groq_model = os.getenv("GROQ_MODEL", "")
    env_hf_model = os.getenv("HUGGINGFACE_MODEL", "")

    # OPTIMIZATION: Pre-calculate Base64 image and LangChain message payload once
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    langchain_msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
    ])

    # 1. Try Google Gemini
    if google_keys:
        # RESTORED: Your 2026 models
        google_models_to_try = [
            'gemini-3.7-flash', 
            'gemini-3.6-flash', 
            'gemini-3.5-flash', 
            'gemini-2.5-flash', 
            'gemini-2.5-pro'
        ]
        if env_g_model:
            google_models_to_try.insert(0, env_g_model)
        
        for i in range(len(google_keys)):
            current_key = next(key_iterator)
            genai.configure(api_key=current_key)
            
            for g_model in google_models_to_try:
                try:
                    print(f"DEBUG: Trying Google Key #{i+1} with {g_model}...", flush=True)
                    model = genai.GenerativeModel(g_model)
                    img = Image.open(io.BytesIO(image_bytes))
                    
                    # ADDED: Timeout so it fails fast if Google is hanging
                    response = model.generate_content(
                        [prompt, img],
                        request_options={"timeout": 10} 
                    )
                    print(f"DEBUG: Google SUCCESS with {g_model}", flush=True)
                    return clean_ai_text(response.text)
                except Exception as e:
                    print(f"DEBUG: Google {g_model} FAILED: {str(e)}", flush=True)
                    continue

    # 2. Try OpenRouter
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        or_models_to_try = ["meta-llama/llama-3.2-11b-vision-instruct", "qwen/qwen-2-vl-72b-instruct"]
        if env_or_model:
            or_models_to_try.insert(0, env_or_model)
        
        for or_model in or_models_to_try:
            try:
                print(f"DEBUG: Trying OpenRouter with {or_model}...", flush=True)
                llm = ChatOpenAI(
                    model=or_model, 
                    api_key=openrouter_key, 
                    base_url="https://openrouter.ai/api/v1",
                    timeout=10 # ADDED: Fail fast
                )
                response = llm.invoke([langchain_msg])
                print(f"DEBUG: OpenRouter SUCCESS with {or_model}", flush=True)
                return clean_ai_text(response.content)
            except Exception as e:
                print(f"DEBUG: OpenRouter {or_model} FAILED: {str(e)}", flush=True)
                continue

    # 3. Try Groq
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        # RESTORED: Your Llama 4 model
        groq_models_to_try = ["meta-llama/llama-4-scout-17b-16e-instruct"]
        if env_groq_model:
            groq_models_to_try.insert(0, env_groq_model)

        for groq_model in groq_models_to_try:
            try:
                print(f"DEBUG: Trying Groq with {groq_model}...", flush=True)
                llm = ChatGroq(
                    model=groq_model, 
                    temperature=0, 
                    api_key=groq_key,
                    timeout=10 # ADDED: Fail fast
                )
                response = llm.invoke([langchain_msg])
                print(f"DEBUG: Groq SUCCESS with {groq_model}", flush=True)
                return clean_ai_text(response.content)
            except Exception as e:
                print(f"DEBUG: Groq {groq_model} FAILED: {str(e)}", flush=True)
                continue

    # 4. Try Hugging Face
    hf_key = os.getenv("HUGGINGFACE_API_KEY")
    if hf_key:
        hf_models_to_try = ["meta-llama/Llama-3.2-11B-Vision-Instruct", "Qwen/Qwen2-VL-7B-Instruct"]
        if env_hf_model:
            hf_models_to_try.insert(0, env_hf_model)

        for hf_model in hf_models_to_try:
            try:
                print(f"DEBUG: Trying Hugging Face with {hf_model}...", flush=True)
                llm = ChatOpenAI(
                    model=hf_model, 
                    api_key=hf_key, 
                    base_url="https://api-inference.huggingface.co/v1/",
                    timeout=10 # ADDED: Fail fast
                )
                response = llm.invoke([langchain_msg])
                print(f"DEBUG: Hugging Face SUCCESS with {hf_model}", flush=True)
                return clean_ai_text(response.content)
            except Exception as e:
                print(f"DEBUG: Hugging Face {hf_model} FAILED: {str(e)}", flush=True)
                continue

    return "Network error. Please contact Zubair for support."

@app.post("/analyze-scene")
async def analyze_scene(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        prompt = "Describe the scene for a blind person. Be conversational and warn of hazards."
        result = call_vision_model(prompt, image_bytes)
        return {"status": "success", "script": result}
    except Exception as e:
        print(f"DEBUG: ENDPOINT ERROR in analyze-scene: {str(e)}", flush=True)
        return {"status": "error", "message": "Connection lost. Please contact Zubair for support."}

@app.post("/ask-vision")
async def ask_vision(image: UploadFile = File(...), question: str = Form(...)):
    try:
        image_bytes = await image.read()
        prompt = f"The user asks: {question}. Tell them where the object is or guide them."
        result = call_vision_model(prompt, image_bytes)
        return {"status": "success", "script": result}
    except Exception as e:
        print(f"DEBUG: ENDPOINT ERROR in ask-vision: {str(e)}", flush=True)
        return {"status": "error", "message": "Connection lost. Please contact Zubair for support."}

@app.post("/navigate")
async def navigate(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        prompt = "Walking guide: 1 short sentence about what is directly ahead and distance."
        result = call_vision_model(prompt, image_bytes)
        return {"status": "success", "script": result}
    except Exception as e:
        print(f"DEBUG: ENDPOINT ERROR in navigate: {str(e)}", flush=True)
        return {"status": "error", "message": "Connection lost. Please contact Zubair for support."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
