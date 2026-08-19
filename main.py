import os
import base64
import re
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from fastapi import FastAPI, UploadFile, File, Form

print("DEBUG: SERVER STARTING - GOOGLE VERTEX AI (2026 MODELS)", flush=True)

app = FastAPI(title="binAI Human Assistant Backend")

# ==========================================
# INITIALIZE GOOGLE CLOUD VERTEX AI
# ==========================================
PROJECT_ID = "project-3160f2ec-9f07-4d03-a9e" 
REGION = "us-central1"

try:
    vertexai.init(project=PROJECT_ID, location=REGION)
    print("DEBUG: Vertex AI Initialized Successfully", flush=True)
except Exception as e:
    print(f"DEBUG: Failed to initialize Vertex AI: {e}", flush=True)

@app.get("/")
@app.head("/")
async def health_check():
    return {"status": "alive", "message": "binAI Backend is running on Vertex AI!"}

def clean_ai_text(raw_text):
    clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL | re.IGNORECASE)
    clean_text = re.sub(r'<thought>.*?</thought>', '', clean_text, flags=re.DOTALL | re.IGNORECASE)
    return clean_text.replace('<', '').replace('>', '').strip()

def call_vision_model(prompt, image_bytes):
    # Updated to the active 2026 models based on official docs
    google_models_to_try = [
        'gemini-2.5-flash',
        'gemini-2.5-pro'
    ]
    
    for g_model in google_models_to_try:
        try:
            print(f"DEBUG: Trying Google Vertex AI with {g_model}...", flush=True)
            
            model = GenerativeModel(g_model)
            image_part = Part.from_data(mime_type="image/jpeg", data=image_bytes)
            
            response = model.generate_content([prompt, image_part])
            
            print(f"DEBUG: Google Vertex SUCCESS with {g_model}", flush=True)
            return clean_ai_text(response.text)
            
        except Exception as e:
            print(f"DEBUG: Google Vertex {g_model} FAILED: {str(e)}", flush=True)
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
        print(f"DEBUG: Endpoint Error: {str(e)}", flush=True)
        return {"status": "error", "message": "Connection lost. Please contact Zubair for support."}

@app.post("/ask-vision")
async def ask_vision(image: UploadFile = File(...), question: str = Form(...)):
    try:
        image_bytes = await image.read()
        prompt = f"The user asks: {question}. Tell them where the object is or guide them."
        result = call_vision_model(prompt, image_bytes)
        return {"status": "success", "script": result}
    except Exception as e:
        print(f"DEBUG: Endpoint Error: {str(e)}", flush=True)
        return {"status": "error", "message": "Connection lost. Please contact Zubair for support."}

@app.post("/navigate")
async def navigate(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        # NEW AGGRESSIVE PROMPT FOR WALK MODE
        prompt = """
        You are a bodyguard guiding a blind person walking forward. 
        Look at the image. 
        1. If there is a wall, obstacle, person, or drop-off VERY CLOSE directly ahead, you MUST reply starting with the word "STOP!" followed by what it is (e.g., "STOP! Wall right in front of you!").
        2. If there is an obstacle slightly further away, warn them briefly (e.g., "Desk 3 feet ahead").
        3. If the path is clear, reply with exactly two words: "Path clear."
        Do not be polite. Be urgent, short, and direct.
        """
        result = call_vision_model(prompt, image_bytes)
        
        # If the AI says "Path clear", we can optionally silence it so it doesn't annoy the user, 
        # or just let it say "Path clear". We will let it speak so the user knows it's working.
        return {"status": "success", "script": result}
    except Exception as e:
        print(f"DEBUG: Endpoint Error: {str(e)}", flush=True)
        return {"status": "error", "message": "Connection lost. Please contact Zubair for support."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
