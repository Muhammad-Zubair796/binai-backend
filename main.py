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

        # 2. The Pro Prompt (Strictly for description)
        prompt_text = """
        Describe this scene for a blind person in 3 simple, friendly sentences. 
        Identify objects, read medicine names, and warn about hazards. 
        Output ONLY the description. Do not mention rules, instructions, or markdown.
        """

        msg = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        )

        # 3. ACTIVE MODEL LIST (Instruct versions)
        # These are the current replacements for the decommissioned preview models.
        vision_models_to_try = [
            "llama-3.2-11b-vision-instruct",
            "llama-3.2-90b-vision-instruct"
        ]
        
        response_text = None
        error_log = ""

        for model_name in vision_models_to_try:
            try:
                llm = ChatGroq(
                    model=model_name, 
                    temperature=0.1,
                    groq_api_key=os.getenv("GROQ_API_KEY")
                )
                response = llm.invoke([msg])
                response_text = response.content.strip()
                if response_text:
                    break
            except Exception as e:
                error_log += f"{model_name}: {str(e)} "
                continue

        if response_text:
            return {"status": "success", "script": response_text}
        else:
            return {"status": "error", "message": f"Models failed. Details: {error_log}"}

    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
