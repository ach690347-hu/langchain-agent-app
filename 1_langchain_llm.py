import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()



llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

response = llm.invoke(
    "Break down the steps required for an AI Agent to autonomously book a flight ticket"
)


print(response.text)