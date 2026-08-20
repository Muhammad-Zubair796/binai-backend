import os
import base64
import re
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional

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
async def analyze_scene(image: UploadFile = File(...), on_device_data: Optional[str] = Form(None)):
    try:
        image_bytes = await image.read()
        
        sensor_injection = ""
        if on_device_data:
            sensor_injection = f"ON-DEVICE SENSOR DATA: {on_device_data}. Use these exact distances and face details in your response instead of guessing."

        prompt = f"""
        You are an expert mobility instructor and visual assistant for a totally blind person. Describe this scene to help them understand their surroundings safely.
        {sensor_injection}
        Structure your response strictly as follows:
        1. Immediate Hazards: Mention any trip hazards, drop-offs, or head-level obstacles first, using the exact sensor distance. If none, skip.
        2. People & Faces: If the sensor data mentions faces (e.g., smiling, eyes open), describe them warmly.
        3. Currency & Items: If the user is holding money, explicitly state the exact denomination and currency.
        4. Scene Overview: Briefly state the environment and mention key objects using clock directions.
        Keep it under 3 short sentences. Be highly spatial and precise.
        """
        result = call_vision_model(prompt, image_bytes)
        return {"status": "success", "script": result}
    except Exception as e:
        print(f"DEBUG: Endpoint Error: {str(e)}", flush=True)
        return {"status": "error", "message": "Connection lost. Please contact Zubair for support."}

@app.post("/ask-vision")
async def ask_vision(image: UploadFile = File(...), question: str = Form(...), on_device_data: Optional[str] = Form(None)):
    try:
        image_bytes = await image.read()
        
        sensor_injection = ""
        if on_device_data:
            sensor_injection = f"ON-DEVICE SENSOR DATA: {on_device_data}. Use these exact distances."

        prompt = f"""
        You are an expert visual assistant for a blind person. The user's command is: "{question}"
        {sensor_injection}
        Look at the image and follow the user's command exactly. 
        - If they are asking to find an object, and you see it, give its exact location using the sensor distance and clock directions.
        - If they are asking to find an object and it is NOT there, follow their exact failure instruction (e.g., saying "NO").
        - If they ask about currency, identify the exact denomination.
        Keep your answer under 2 sentences. Be precise, spatial, and highly practical.
        """
        result = call_vision_model(prompt, image_bytes)
        return {"status": "success", "script": result}
    except Exception as e:
        print(f"DEBUG: Endpoint Error: {str(e)}", flush=True)
        return {"status": "error", "message": "Connection lost. Please contact Zubair for support."}

@app.post("/navigate")
async def navigate(image: UploadFile = File(...), previous_context: Optional[str] = Form(None), on_device_data: Optional[str] = Form(None)):
    try:
        image_bytes = await image.read()
        
        memory_instruction = ""
        if previous_context and previous_context != "Path clear.":
            memory_instruction = f"""
            MEMORY ALERT: 2 seconds ago, you warned the user: "{previous_context}". 
            Compare the current image to your previous warning. If that object is getting closer/larger, it is moving toward the user! You MUST yell "STOP! [Object] approaching fast!"
            """
            
        sensor_injection = ""
        if on_device_data:
            sensor_injection = f"ON-DEVICE SENSOR DATA: {on_device_data}. Use these exact distances for your evasion commands."

        prompt = f"""
        You are a real-time mobility radar for a blind person walking forward. Analyze the immediate path ahead.
        {memory_instruction}
        {sensor_injection}
        You MUST provide a highly urgent, ultra-short response (MAXIMUM 8 WORDS) to prevent injury.
        Rules:
        1. CRITICAL DANGER (Stairs, drop-offs, moving cars): Start with "STOP!" followed by the danger (e.g., "STOP! Stairs going down", "STOP! Car approaching").
        2. BLOCKED PATH: State the object, exact sensor distance, and an evasion command (e.g., "Wall 3 feet, move right", "Person 2 feet, shift left").
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
