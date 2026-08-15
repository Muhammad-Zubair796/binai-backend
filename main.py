from fastapi import FastAPI, UploadFile, File
import shutil
import os
import base64
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Initialize FastAPI app
app = FastAPI(title="binAI Backend")

@app.get("/")
async def health_check():
    return {"status": "alive", "message": "binAI Backend is running!"}

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

# --- Define Agents ---
vision_analyst = Agent(
    role='Senior Vision Analyst',
    goal='Extract every detail from visual data provided',
    backstory='Expert in computer vision, trained to identify medical labels and hazards.',
    llm=llm,
    allow_delegation=False,
    verbose=True
)

pharmacist = Agent(
    role='Clinical Pharmacist',
    goal='Identify medicine and provide safe dosage instructions',
    backstory='Specialized in pharmaceutical labeling and patient safety.',
    llm=llm,
    allow_delegation=False,
    verbose=True
)

safety_officer = Agent(
    role='Safety & Navigation Specialist',
    goal='Identify hazards and provide directional guidance',
    backstory='Expert in mobility for the visually impaired.',
    llm=llm,
    allow_delegation=False,
    verbose=True
)

narrator = Agent(
    role='Lead Patient Coordinator',
    goal='Synthesize reports into a friendly voice message',
    backstory='Trained in patient communication, ensuring clarity and comfort.',
    llm=llm,
    allow_delegation=False,
    verbose=True
)

@app.post("/analyze-scene")
async def analyze_scene(image: UploadFile = File(...)):
    try:
        # 1. Save Image
        temp_image_path = f"temp_{image.filename}"
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # 2. Convert Image to Base64 for LangChain
        with open(temp_image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        # 3. Initial Vision Extraction (Fixed for LangChain)
        vision_prompt = "Describe this image in extreme detail for a blind person. List all text, medicine names, expiry dates, and any physical obstacles like stairs or sharp objects."
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": vision_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}}
            ]
        )
        
        vision_report = llm.invoke([message]).content

        # 4. Define Agentic Tasks
        task1 = Task(
            description=f"Review this raw vision report: {vision_report}. Categorize all identified items into Medicine, Documents, or Hazards.",
            agent=vision_analyst,
            expected_output="A categorized list of everything in the scene."
        )
        task2 = Task(
            description="Review the categorized list. If medicine is found, provide clear dosage and safety warnings. If not, ignore.",
            agent=pharmacist,
            expected_output="Clinical safety advice."
        )
        task3 = Task(
            description="Review the categorized list. Identify immediate physical dangers and give directional advice (e.g., 'Step back').",
            agent=safety_officer,
            expected_output="Safety and navigation warnings."
        )
        task4 = Task(
            description="Synthesize the medical and safety advice into a calm, friendly 30-second script for the patient. No markdown, just plain text.",
            agent=narrator,
            expected_output="A final spoken script."
        )

        # 5. Run the Crew
        binAI_crew = Crew(
            agents=[vision_analyst, pharmacist, safety_officer, narrator],
            tasks=[task1, task2, task3, task4],
            process=Process.sequential
        )

        result = binAI_crew.kickoff()

        # Clean up
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

        return {"status": "success", "script": str(result)}

    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
