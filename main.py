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
        # PROMPT UPDATE: Added distance estimation and clock-face directions.
        prompt = """
        You are an expert mobility instructor for a totally blind person. Describe this scene to help them understand their surroundings safely.
        Structure your response strictly as follows:
        1. Immediate Hazards: Mention any trip hazards, drop-offs, or head-level obstacles first, estimating distance (e.g., "Trip hazard 2 feet ahead"). If none, skip.
        2. Scene Overview: Briefly state the environment (e.g., "You are in a busy hallway").
        3. Key Objects: Mention locations using clock directions and estimated distances (e.g., "Desk 5 feet away at 12 o'clock", "Door at 3 o'clock").
        Keep it under 3 short sentences. Be highly spatial and precise.
        """
        result = call_vision_model(prompt, image_bytes)
        return {"status": "success", "script": result}
    except Exception as e:
        print(f"DEBUG: Endpoint Error: {str(e)}", flush=True)
        return {"status": "error", "message": "Connection lost. Please contact Zubair for support."}

@app.post("/ask-vision")
async def ask_vision(image: UploadFile = File(...), question: str = Form(...)):
    try:
        image_bytes = await image.read()
        # PROMPT UPDATE: Forces exact distance and directional guidance.
        prompt = f"""
        You are an expert visual assistant for a blind person. The user asks: "{question}"
        Look at the image and answer directly. 
        - If the object is present, give its exact location using estimated distance and directions (e.g., "It is about 3 feet away, slightly to your left on the table").
        - If the object is NOT in the image, clearly state: "I do not see that in the current view."
        - Keep your answer under 2 sentences. Be precise, spatial, and highly practical.
        """
        result = call_vision_model(prompt, image_bytes)
        return {"status": "success", "script": result}
    except Exception as e:
        print(f"DEBUG: Endpoint Error: {str(e)}", flush=True)
        return {"status": "error", "message": "Connection lost. Please contact Zubair for support."}

@app.post("/navigate")
async def navigate(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        # PROMPT UPDATE: Ultra-short, military-style radar. Forces distance and evasion commands.
        prompt = """
        You are a real-time mobility radar for a blind person walking forward. Analyze the immediate path ahead.
        You MUST provide a highly urgent, ultra-short response (MAXIMUM 6 WORDS) to prevent injury.
        Rules:
        1. CRITICAL DANGER (Stairs, drop-offs, moving cars): Start with "STOP!" followed by the danger (e.g., "STOP! Stairs going down", "STOP! Car approaching").
        2. BLOCKED PATH: State the object, estimated distance, and an evasion command (e.g., "Wall 3 feet, move right", "Person 2 steps, shift left", "Pole 1 foot, step right").
        3. CLEAR PATH: If the immediate walking path is completely clear for at least 10 feet, reply exactly: "Path clear."
        Do NOT use full sentences. Do NOT be polite. Prioritize distance (feet/steps) and directional commands (left/right).
        """
        result = call_vision_model(prompt, image_bytes)
        
        return {"status": "success", "script": result}
    except Exception as e:
        print(f"DEBUG: Endpoint Error: {str(e)}", flush=True)
        return {"status": "error", "message": "Connection lost. Please contact Zubair for support."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
