import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import TypedDict

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

def planner_agent(state, llm):
    prompt = f""" 
    A user wants to study :{state['user_input']}

    Break this into key learning topics.
    Return only a list of topics
    """

    response = llm.invoke(prompt)

    state["topics"] = response.content
    return state


def resource_agent(state, llm):

    prompt = f"""
    Topics: 
    {state['topics']}

    Find study resources for these topics.
    """

    response = llm.invoke(prompt)
    state["resources"] = response.content
    return state


def schedule_agent(state, llm):
    prompt = f"""
    Topics:
    {state['topics']}

    Create a day-by-day schedule for these topics
    """

    response = llm.invoke(prompt)
    state["schedule"] = response.content

    return state


def reviewer_agent(state, llm):
    prompt = f"""
    Study Schedule:
    {state['schedule']}

    Resources:
    {state['resources']}
    """

    response = llm.invoke(prompt)
    state["final_plan"] = response.content
    return state
    

def main():
    user_input = input("What do you want to learn? ")
    state = {
        "user_input": user_input,
        "topics": "",
        "resources": "",
        "schedule": "",
        "final_plan": ""
    }

    state = planner_agent(state, llm)
    state = resource_agent(state, llm)
    state = schedule_agent(state, llm)
    state = reviewer_agent(state, llm)

    print("\nFinal Study Plan:\n")
    print(state["final_plan"])

if __name__ == "__main__":
    main()