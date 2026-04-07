import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import TypedDict
from langgraph.graph import StateGraph, END
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

class AgentState(TypedDict):
    user_input: str
    topics:str
    resources:str
    schedule:str
    final_plan:str

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2

)

def planner_agent(state):
    print("\nRunning Planner Agent...")
    prompt = f""" 
    you are a learning planner
    A user wants to study :{state['user_input']}

    Break this into key learning topics.

    Rules:
    - Return only numbered list
    - Keep topics concise
    - 5-8 topics maximum
    - Do not use markdown bold (**) anywhere.

    """

    response = llm.invoke(prompt)

    return {"topics": response.content}



def resource_agent(state):
    print("\nRunning Resource Agent...")
    prompt = f"""
    Topics: 
    {state['topics']}

    Suggest good learning resources

    Rules:
    - Provide 2-3 resources per topic
    - Prefer official documentation, tutorials, and well-known platforms
    - Keep answers short
    - Do not use markdown bold (**) anywhere.

    """

    response = llm.invoke(prompt)
    return {"resources": response.content}


def schedule_agent(state):

    print("\nRunning Schedule Agent...")
    prompt = f"""
    Topics:
    {state['topics']}

    Create a daily learning schedule

    Rules:
    - Use Day 1, Day2 format
    - Cover all topics
    - Keep tasks short
    - Do not use markdown bold (**) anywhere.

    """

    response = llm.invoke(prompt)
    return {"schedule": response.content}


def reviewer_agent(state):
    print("\nRunning Reviewer Agent...")
    prompt = f"""

    Create a final study plan.
    Topics:
    {state['topics']}

    Schedule:
    {state['schedule']}

    Resources:
    {state['resources']}
    Format the output clearly

    Structure:
    Title
    Daily Schedule
    Learning Resources

    Make it easy to read and understand
    - Do not use markdown bold (**) anywhere.

    """

    response = llm.invoke(prompt)
    return {"final_plan": response.content}


workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_agent)
workflow.add_node("resource", resource_agent)
workflow.add_node("schedule", schedule_agent)
workflow.add_node("reviewer", reviewer_agent)

workflow.add_edge("planner", "resource")
workflow.add_edge("resource", "schedule")
workflow.add_edge("schedule", "reviewer")
workflow.add_edge("reviewer", END)

workflow.set_entry_point("planner")

# Rename consolidated app to avoid conflict with FastAPI app
planner_workflow = workflow.compile()

# FastAPI setup
app = FastAPI(title="AI Study Planner API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GoalRequest(BaseModel):
    goal: str

@app.post("/generate")
async def generate_plan(request: GoalRequest):
    try:
        state = {
            "user_input": request.goal,
            "topics": "",
            "resources": "",
            "schedule": "",
            "final_plan": ""
        }
        result = planner_workflow.invoke(state)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    # CLI fallback
    user_input = input("What do you want to learn? ")
    state = {
        "user_input": user_input,
        "topics": "",
        "resources": "",
        "schedule": "",
        "final_plan": ""
    }

    result = planner_workflow.invoke(state)
    print("\n==============================")
    print("FINAL STUDY PLAN")
    print("==============================\n")
    print(result["final_plan"])

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        main()