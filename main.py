import os
import shutil
import PIL.Image
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File

# Initialize FastAPI app
app = FastAPI(title="binAI Backend Fast")

# Configure Gemini directly (Lightning Fast)
# Ensure GOOGLE_API_KEY is set in Render Environment Variables
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

@app.get("/")
async def health_check():
    return {"status": "alive", "message": "binAI Fast Backend is running!"}

@app.post("/analyze-scene")
async def analyze_scene(image: UploadFile = File(...)):
    try:
        # 1. Save Image temporarily
        temp_image_path = f"temp_{image.filename}"
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        raw_image = PIL.Image.open(temp_image_path)

        # 2. Single, powerful prompt (Replaces the 4 agents)
        prompt = """
        You are an AI assistant acting as the eyes for a visually impaired person. 
        Look at this image and provide a short, friendly, and clear spoken script (maximum 3 to 4 sentences).
        
        Rules:
        1. Tell them exactly what is in front of them.
        2. If there is medicine, read the name and dosage instructions clearly.
        3. If there is a physical hazard (stairs, sharp objects, obstacles), warn them immediately.
        4. DO NOT use markdown, asterisks, or bullet points. Write it exactly as it should be spoken out loud by a Text-to-Speech engine.
        """

        # 3. Call Gemini directly
        response = model.generate_content([prompt, raw_image])

        # Clean up
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

        # Return the text immediately
        return {"status": "success", "script": response.text.strip()}

    except Exception as e:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
