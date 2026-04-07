import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import TypedDict
from langgraph.graph import StateGraph, END

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
    prompt = f""" 
    A user wants to study :{state['user_input']}

    Break this into key learning topics.
    """

    response = llm.invoke(prompt)

    return {"topics": response.content}


def resource_agent(state):

    prompt = f"""
    Topics: 
    {state['topics']}

    Find study resources for these topics.
    """

    response = llm.invoke(prompt)
    return {"resources": response.content}


def schedule_agent(state):
    prompt = f"""
    Topics:
    {state['topics']}

    Create a day-by-day schedule for these topics
    """

    response = llm.invoke(prompt)
    return {"schedule": response.content}


def reviewer_agent(state):
    prompt = f"""
    Study Schedule:
    {state['schedule']}

    Resources:
    {state['resources']}
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

app = workflow.compile()
    

def main():
    user_input = input("What do you want to learn? ")
    state = {
        "user_input": user_input,
        "topics": "",
        "resources": "",
        "schedule": "",
        "final_plan": ""
    }

    result = app.invoke(state)

    print("\nFinal Study Plan:\n")
    print(result["final_plan"])

if __name__ == "__main__":
    main()