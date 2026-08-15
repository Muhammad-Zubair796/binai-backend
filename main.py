from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os

# Your CrewAI imports
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize FastAPI app
app = FastAPI(title="binAI Backend", description="AI Medical & Safety Assistant for the Blind")

# Initialize the high-end LLM (Make sure your GOOGLE_API_KEY is set in your environment)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

# --- Define Agents ---
vision_analyst = Agent(
    role='Senior Vision Analyst',
    goal='Extract every detail from the visual data provided',
    backstory='Expert in computer vision and OCR, trained to see what others miss.',
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
    goal='Synthesize all agent reports into a single, empathetic voice message',
    backstory='Trained in patient communication, ensuring clarity and comfort.',
    llm=llm,
    allow_delegation=False, # Changed to False to prevent infinite loops in simple setups
    verbose=True
)

# --- Define the API Endpoint ---
@app.post("/analyze-scene")
async def analyze_scene(image: UploadFile = File(...)):
    """
    This endpoint receives an image from the Android app, 
    runs the CrewAI agents, and returns the spoken script.
    """
    try:
        # 1. Save the uploaded image temporarily so the AI can read it
        temp_image_path = f"temp_{image.filename}"
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # 2. Define Tasks (Notice we pass the image path to the vision analyst)
        task1 = Task(
            description=f"Analyze the image located at this file path: {temp_image_path}. List all text, medicine labels, and physical objects in the scene.", 
            agent=vision_analyst,
            expected_output="A detailed text description of all objects, text, and potential hazards in the image."
        )
        task2 = Task(
            description="Based on the vision analyst's report, if medicine is present, verify dosage and safety. If no medicine is present, state that.", 
            agent=pharmacist,
            expected_output="Medical safety advice and dosage instructions."
        )
        task3 = Task(
            description="Based on the vision analyst's report, identify any immediate physical hazards (e.g., edges, sharp objects, hot liquids).", 
            agent=safety_officer,
            expected_output="A list of physical hazards and navigation warnings."
        )
        task4 = Task(
            description="Take the reports from the pharmacist and safety officer. Create a final, friendly, 30-second audio script to be read to the blind patient. Do not include markdown formatting, just the spoken words.", 
            agent=narrator,
            expected_output="A plain text script ready for Text-to-Speech."
        )

        # 3. Create and Run the Crew
        binAI_crew = Crew(
            agents=[vision_analyst, pharmacist, safety_officer, narrator],
            tasks=[task1, task2, task3, task4],
            process=Process.sequential
        )

        # Kickoff the AI process
        result = binAI_crew.kickoff()

        # 4. Clean up (delete the temporary image)
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

        # 5. Return the result to the Android App
        # Note: We return text. The Android app will handle turning this text into speech!
        return {"status": "success", "script": str(result)}

    except Exception as e:
        return {"status": "error", "message": str(e)}
