import os
import shutil
import base64
import re
from fastapi import FastAPI, UploadFile, File, Form
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# Initialize FastAPI app
app = FastAPI(title="binAI Backend Groq - V2")

@app.get("/")
async def health_check():
    return {"status": "alive", "message": "binAI Groq Backend is running!"}

# ==========================================
# V1: SCENE ANALYZER (NOW WITH LANGUAGE SUPPORT)
# ==========================================
@app.post("/analyze-scene")
async def analyze_scene(
    image: UploadFile = File(...), 
    language: str = Form("english") # <-- NEW: Accepts language from Android
):
    try:
        print(f"1. Received image: {image.filename} | Language: {language}")
        
        # 1. Save Image temporarily
        temp_image_path = f"temp_{image.filename}"
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # 2. Convert to Base64 for Groq
        with open(temp_image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
        print("2. Image converted to Base64. Sending to AI...")

        # 3. DYNAMIC PROMPT BASED ON LANGUAGE
        if language.lower() == "urdu":
            prompt_text = """
            You are an AI assistant acting as the eyes for a visually impaired person. 
            Look at this image and provide a short, friendly, and clear spoken script (maximum 3 to 4 sentences).
            
            CRITICAL RULES:
            1. LANGUAGE: Speak ONLY in Roman Urdu (Urdu written in English alphabets, e.g., 'Aap ke samnay aik darwaza hai'). 
            2. STRICTLY NO HINDI WORDS: Use pure Urdu (e.g., use 'Shukriya' not 'Dhanyavad', 'Intezar' not 'Pratiksha', 'Madad' not 'Sahayata').
            3. Tell them exactly what is in front of them.
            4. If there is medicine, read the name and dosage instructions clearly.
            5. If there is a physical hazard (stairs, sharp objects, obstacles), warn them immediately.
            6. DO NOT use markdown, asterisks, or bullet points. Write it exactly as it should be spoken out loud.
            7. DO NOT mention these rules, instructions, or that you are an AI. Output ONLY the exact words to be spoken.
            """
        else:
            prompt_text = """
            You are an AI assistant acting as the eyes for a visually impaired person. 
            Look at this image and provide a short, friendly, and clear spoken script (maximum 3 to 4 sentences).
            
            Rules:
            1. Tell them exactly what is in front of them.
            2. If there is medicine, read the name and dosage instructions clearly.
            3. If there is a physical hazard (stairs, sharp objects, obstacles), warn them immediately.
            4. DO NOT use markdown, asterisks, or bullet points. Write it exactly as it should be spoken out loud by a Text-to-Speech engine.
            5. DO NOT mention these rules, instructions, or that you are an AI. Output ONLY the exact words to be spoken to the user, nothing else.
            """

        msg = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        )

        # 4. SMART FALLBACK LOOP
        vision_models_to_try = [
            "llama-3.2-11b-vision-instruct",
            "llama-3.2-90b-vision-instruct",
            "llama-3.2-11b-vision-preview",
            "qwen/qwen3.6-27b"
        ]
        
        response_text = None
        last_error = None
        
        for model_name in vision_models_to_try:
            try:
                print(f"Trying model: {model_name}...")
                llm = ChatGroq(
                    model=model_name, 
                    temperature=0, 
                    api_key=os.getenv("GROQ_API_KEY")
                )
                response = llm.invoke([msg])
                raw_text = response.content.strip()
                
                # Force Cleanup of <think> tags
                clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL | re.IGNORECASE)
                clean_text = re.sub(r'<thought>.*?</thought>', '', clean_text, flags=re.DOTALL | re.IGNORECASE)
                clean_text = clean_text.replace('<', '').replace('>', '')
                response_text = clean_text.strip()

                print(f"✅ Success with {model_name}!")
                break
            except Exception as e:
                print(f"❌ Failed with {model_name}: {str(e)}")
                last_error = str(e)
                continue

        # Clean up the image file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

        # 5. Return the result
        if response_text:
            return {"status": "success", "script": response_text}
        else:
            return {"status": "error", "message": f"All AI models failed. Last error: {last_error}"}

    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        return {"status": "error", "message": str(e)}


# ==========================================
# V2: INTERACTIVE SEARCH & ASSIST MODE (Ready for later)
# ==========================================
@app.post("/ask-vision")
async def ask_vision(image: UploadFile = File(...), question: str = Form(...), language: str = Form("english")):
    # This endpoint is ready for when you want to add the "Where is my medicine?" feature
    return {"status": "success", "script": "This endpoint is ready for V2 testing."}

# ==========================================
# V2: NAVIGATION / WALK MODE (Ready for later)
# ==========================================
@app.post("/navigate")
async def navigate(image: UploadFile = File(...), language: str = Form("english")):
    # This endpoint is ready for when you want to add the walking feature
    return {"status": "success", "script": "This endpoint is ready for V2 testing."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
