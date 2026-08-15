import os
import shutil
import PIL.Image
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File

app = FastAPI(title="binAI Backend Fast")

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

@app.get("/")
async def health_check():
    return {"status": "alive", "message": "binAI Fast Backend is running!"}

@app.post("/analyze-scene")
async def analyze_scene(image: UploadFile = File(...)):
    try:
        print(f"1. Received image: {image.filename}")
        temp_image_path = f"temp_{image.filename}"
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        raw_image = PIL.Image.open(temp_image_path)
        print("2. Image loaded successfully. Sending to Gemini...")

        prompt = """
        You are an AI assistant acting as the eyes for a visually impaired person. 
        Look at this image and provide a short, friendly, and clear spoken script (maximum 3 to 4 sentences).
        
        Rules:
        1. Tell them exactly what is in front of them.
        2. If there is medicine, read the name and dosage instructions clearly.
        3. If there is a physical hazard (stairs, sharp objects, obstacles), warn them immediately.
        4. DO NOT use markdown, asterisks, or bullet points. Write it exactly as it should be spoken out loud by a Text-to-Speech engine.
        """

        response = model.generate_content([prompt, raw_image])
        print("3. Gemini response received!")

        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

        return {"status": "success", "script": response.text.strip()}

    except Exception as e:
        print(f"ERROR: {str(e)}")
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
