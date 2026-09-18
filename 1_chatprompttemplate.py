# import os
# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate

# load_dotenv()

# llm = ChatGoogleGenerativeAI(
#     model="gemini-3.6-flash",
#     temperature=0.7,
#     google_api_key=os.getenv("GOOGLE_API_KEY")
# )

# chat_template = ChatPromptTemplate.from_messages([
#     ("system", "you are a mathematical teacher. guide the maths concepts..."),
#     ("human", "Tell me a multiplication of {value1} and {value2}.")
# ])

# messages = chat_template.format_messages(value1=24, value2=97)

# print("----- Generated Messages -----")
# for m in messages:
   
#     print(f"{m.type}: {m.content}")

# response = llm.invoke(messages)

# print("\n----- LLM Response -----")
# print(response.text)


import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly restaurant guide who gives short and simple suggestions."),
    ("human", "Suggest 3 good {food} restaurants in {city}.")
])

messages = chat_template.format_messages(food="punjabi", city="Lahore")


print("----- Generated Messages -----")
for m in messages:
    print(f"{m.type}: {m.content}")


response = llm.invoke(messages)

print("\n----- LLM Response -----")
print(response.text)