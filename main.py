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
        # PROMPT UPDATE: Structured, hazard-first, concise description.
        prompt = """
        You are an expert AI visual assistant for a visually impaired user. Analyze this image and provide a clear, concise, and highly useful description.
        Follow this strict structure:
        1. Immediate Hazards: Mention any obstacles, drop-offs, or dangers right in front of the user first.
        2. Main Subject: Describe the primary objects or people in the scene.
        3. Environment: Briefly state the setting (e.g., 'indoor office', 'busy street').
        Keep the response under 3 sentences. Be direct, professional, and avoid filler words like 'I can see' or 'The image shows'.
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
        # PROMPT UPDATE: Forces spatial awareness and strict fallback if not found.
        prompt = f"""
        You are an expert AI visual assistant for a visually impaired user. The user is asking: "{question}"
        Look at the image and answer directly. 
        - If the object is present, describe its exact location using relative directions (e.g., 'in the center', 'slightly to your left', 'on the table in front of you').
        - If the object is NOT in the image, clearly state: "I do not see that in the current view."
        - Keep your answer concise, accurate, and highly practical for someone who cannot see. Do not use filler words.
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
        # PROMPT UPDATE: Ultra-fast, safety-critical radar mode for 2-second intervals.
        prompt = """
        You are a highly critical safety radar for a blind person walking forward. Analyze the immediate path ahead (the center of the image).
        Rules for your response:
        1. CRITICAL DANGER: If there is a wall, vehicle, person, stairs, or obstacle VERY CLOSE directly in the path, reply starting with "STOP!" followed by a 2-3 word description (e.g., "STOP! Wall ahead", "STOP! Stairs going down").
        2. UPCOMING OBSTACLE: If there is an obstacle in the path but slightly further away, give a short warning (e.g., "Chair ahead on the right", "Person approaching").
        3. CLEAR PATH: If the immediate walking path is completely clear of obstacles, reply with exactly: "Path clear."
        Do not be conversational. Be urgent, extremely brief, and prioritize physical safety above all else.
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
