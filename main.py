import os
import base64
import io
from fastapi import FastAPI, UploadFile, File
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import PIL.Image
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="binAI Pro Backend")

@app.get("/")
async def health_check():
    return {"status": "alive"}

@app.post("/analyze-scene")
async def analyze_scene(image: UploadFile = File(...)):
    try:
        # 1. Process Image
        image_bytes = await image.read()
        img = PIL.Image.open(io.BytesIO(image_bytes))
        img.thumbnail((800, 800))
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # 2. The Prompt (Simplified to prevent errors)
        prompt_text = "You are the eyes for a blind person. Describe the scene in 3 simple sentences. Warn about hazards or read medicine names if seen. Do not use lists, rules, or markdown. Speak naturally."

        msg = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        )

        # 3. Updated Model List (Newest Groq IDs)
        vision_models_to_try = [
            "llama-3.2-11b-vision-preview",
            "llama-3.2-90b-vision-preview",
            "llava-v1.5-7b-4096-preview"
        ]
        
        response_text = None
        error_details = ""

        for model_name in vision_models_to_try:
            try:
                print(f"Attempting {model_name}...")
                llm = ChatGroq(
                    model=model_name, 
                    temperature=0.2,
                    groq_api_key=os.getenv("GROQ_API_KEY")
                )
                response = llm.invoke([msg])
                response_text = response.content.strip()
                
                if response_text:
                    print(f"✅ Success with {model_name}")
                    break
            except Exception as e:
                print(f"❌ {model_name} failed: {str(e)}")
                error_details += f"{model_name}: {str(e)}. "
                continue

        if response_text:
            return {"status": "success", "script": response_text}
        else:
            # This will show you the REAL error in the JSON response
            return {"status": "error", "message": f"AI Error: {error_details}"}

    except Exception as e:
        return {"status": "error", "message": f"System Error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
