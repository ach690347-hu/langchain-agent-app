import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_community.utilities import GoogleSerperAPIWrapper
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


# Google Search
search = GoogleSerperAPIWrapper(
    serper_api_key=os.getenv("SERPER_API_KEY")
)


system_prompt = """
You are a Google Search Agent.
You can search Google to answer user's questions.
"""


# Memory
memory = MemorySaver()


# Agent
agent = create_agent(
    model=llm,
    tools=[search.run],
    system_prompt=system_prompt,
    checkpointer=memory
)


# =========================
# THREAD 1
# =========================

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "My name is Ayesha."
            }
        ]
    },
    {
        "configurable": {
            "thread_id": "aisha_123"
        }
    }
)

print("\nAI:", response["messages"][-1].content)


# # Same Thread ID
# response = agent.invoke(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "What is my name?"
#             }
#         ]
#     },
#     {
#         "configurable": {
#             "thread_id": "aisha_123"
#         }
#     }
# )

# print("\nAI:", response["messages"][-1].text)


# =========================
# THREAD 2
# =========================

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is my name?"
            }
        ]
    },
    {
        "configurable": {
            "thread_id": "aisha234"
        }
    }
)

print("\nAI:", response["messages"][-1].text)
