from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing_extensions import TypedDict



llm = ChatOpenAI(model="gpt-4o-mini", api_key = "")



# class State(TypedDict):
#     messages: Annotated[list, add_messages]

# graph_builder = StateGraph(State)

# def chatbot(state: State):
#     return {"messages": [llm.invoke(state["messages"])]}

# graph_builder.add_node("chatbot", chatbot)
# graph_builder.add_edge(START, "chatbot")
# graph_builder.add_edge("chatbot", END)

# graph = graph_builder.compile()

# user_input = input("Enter a message: ")

# state = graph.invoke({"messages": [{"role" : "user", "content" : user_input}]})

# print(f"this is the state: {state}")
# print(state["messages"])
# print(f"Ai response: {state["messages"][-1].content}")


# A complex langgraph multi agent setup


class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(..., description = "Classify if the message requires an emotional or logical response")

class StateNew(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None


def classify_message(state: StateNew):
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)

    result = classifier_llm.invoke([
        {
            "role":"system", 
            "content": "You are a message classifier. You are given a message and you need to classify it into an emotional or logical query."
        },
        {
            "role":"user",
            "content": last_message.content
        }
    ])

    return {"message_type": result.message_type}


def router(state: StateNew):
    message_type = state.get("message_type", "logical")
    if message_type == "emotional":
        return {"next": "therapist"}
    
    return {"next": "logical"}


def therapist_agent(state: StateNew):
    last_message = state["messages"][-1]
    messages = [{
        "role": "system",   
        "content": "You are a therapist. You are given a message and you need to respond to it in a way that is helpful and supportive."
    },
    {
        "role": "user",
        "content": last_message.content
    }]

    response = llm.invoke(messages)
    return {"messages": [{
        "role": "assistant",
        "content": response.content
    }]}

def logical_agent(state: StateNew):
    last_message = state["messages"][-1]
    messages = [{
        "role": "system",   
        "content": "You are a logical agent. You are given a message and you need to respond to it in a way that is helpful and supportive."
    },
    {
        "role": "user",
        "content": last_message.content
    }]

    response = llm.invoke(messages)
    return {"messages": [{
        "role": "assistant",
        "content": response.content
    }]}




graph_builder_new = StateGraph(StateNew)

graph_builder_new.add_node("classifier", classify_message)
graph_builder_new.add_node("router", router)
graph_builder_new.add_node("therapist", therapist_agent)
graph_builder_new.add_node("logical", logical_agent)

graph_builder_new.add_edge(START, "classifier")
graph_builder_new.add_edge("classifier", "router")

graph_builder_new.add_conditional_edges("router", lambda state: state.get("next"), {"therapist": "therapist", "logical": "logical"})
graph_builder_new.add_edge("therapist", END)
graph_builder_new.add_edge("logical", END)

graph_new = graph_builder_new.compile()

def run_chatbot():
    state = {"messages": [], "message_type": None}
    while True:
        user_input = input("Enter a message: ")
        if user_input.lower() == "exit":
            print("Exiting chatbot...")
            break
        state['messages'] = state.get("messages", []) + [{"role": "user", "content": user_input}]
        state = graph_new.invoke(state)

        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print(f"AI response: {last_message.content}")

run_chatbot()
