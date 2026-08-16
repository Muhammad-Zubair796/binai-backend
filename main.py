import os
import shutil
import base64
import io
from fastapi import FastAPI, UploadFile, File
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import PIL.Image

app = FastAPI(title="binAI Pro Backend")

@app.get("/")
async def health_check():
    return {"status": "alive"}

@app.post("/analyze-scene")
async def analyze_scene(image: UploadFile = File(...)):
    try:
        # 1. Process Image in memory (Faster & Cleaner)
        image_bytes = await image.read()
        img = PIL.Image.open(io.BytesIO(image_bytes))
        img.thumbnail((800, 800))
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # 2. THE PRO PROMPT
        # We removed the word "Rules" and "1, 2, 3" because they confuse the AI.
        # We added a strict command at the end.
        prompt_text = """
        You are the eyes for a visually impaired person. Describe the scene in front of them.
        
        Provide a natural, friendly description in 3 simple sentences. 
        Identify the main objects, mention any medicine names/dosages if visible, 
        and immediately warn about any hazards like stairs or obstacles.
        
        IMPORTANT: Provide ONLY the spoken description. Do not say 'Here is the description', 
        do not repeat these instructions, and do not use any bullet points or formatting.
        """

        msg = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        )

        # 3. Model Loop
        vision_models_to_try = [
            "llama-3.2-11b-vision-preview", 
            "llama-3.2-90b-vision-preview",
            "llava-v1.5-7b-4096-preview"
        ]
        
        response_text = None
        for model_name in vision_models_to_try:
            try:
                llm = ChatGroq(
                    model=model_name, 
                    temperature=0.2, # Increased slightly for more natural speech
                    groq_api_key=os.getenv("GROQ_API_KEY")
                )
                response = llm.invoke([msg])
                response_text = response.content.strip()
                
                # PRO CLEANUP: If the AI still hallucinates and mentions "Rules" or "Instructions", 
                # we strip them out manually.
                if "Rules:" in response_text or "1." in response_text:
                    continue # Try next model if this one failed to follow instructions
                
                break 
            except:
                continue

        if response_text:
            # Final safety check to ensure it doesn't read the prompt back
            if len(response_text) > 500: # Descriptions should be short
                response_text = "I see a scene in front of you, but I am having trouble describing it clearly. Please try again."
            
            return {"status": "success", "script": response_text}
        else:
            return {"status": "error", "message": "AI is currently unavailable."}

    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
