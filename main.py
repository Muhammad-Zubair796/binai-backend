import os
import shutil
import base64
from fastapi import FastAPI, UploadFile, File
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# Initialize FastAPI app
app = FastAPI(title="binAI Backend Groq")

# Initialize Groq Vision Model
# Make sure you add GROQ_API_KEY to your Render Environment Variables!
llm = ChatGroq(
    model="llama-3.2-11b-vision-preview", 
    temperature=0, 
    api_key=os.getenv("GROQ_API_KEY")
)

@app.get("/")
async def health_check():
    return {"status": "alive", "message": "binAI Groq Backend is running!"}

@app.post("/analyze-scene")
async def analyze_scene(image: UploadFile = File(...)):
    try:
        print(f"1. Received image: {image.filename}")
        
        # 1. Save Image temporarily
        temp_image_path = f"temp_{image.filename}"
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # 2. Convert to Base64 for Groq
        with open(temp_image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
        print("2. Image converted to Base64. Sending to Groq...")

        # 3. Single, powerful prompt for the blind user
        prompt_text = """
        You are an AI assistant acting as the eyes for a visually impaired person. 
        Look at this image and provide a short, friendly, and clear spoken script (maximum 3 to 4 sentences).
        
        Rules:
        1. Tell them exactly what is in front of them.
        2. If there is medicine, read the name and dosage instructions clearly.
        3. If there is a physical hazard (stairs, sharp objects, obstacles), warn them immediately.
        4. DO NOT use markdown, asterisks, or bullet points. Write it exactly as it should be spoken out loud by a Text-to-Speech engine.
        """

        # 4. Construct the LangChain Message
        msg = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        )

        # 5. Call Groq
        response = llm.invoke([msg])
        print("3. Groq response received!")

        # Clean up
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

        # Return the text immediately
        return {"status": "success", "script": response.content.strip()}

    except Exception as e:
        print(f"ERROR: {str(e)}")
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
