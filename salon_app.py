# import os
# import streamlit as st
# from dotenv import load_dotenv

# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.messages import SystemMessage, HumanMessage

# load_dotenv()

# st.set_page_config(page_title="Glam Studio Assistant", page_icon="💄")
# st.title("💄 Glam Studio Beauty Salon")
# st.caption("Ask about services, prices, staff, timings, or book an appointment.")

# llm = ChatGoogleGenerativeAI(
#     model="gemini-3.6-flash",
#     temperature=0.7,
#     google_api_key=os.getenv("GOOGLE_API_KEY")
# )

# system_prompt = """
#                 # Glam Studio Beauty Salon AI Assistant

# You are an AI receptionist for **Glam Studio** — a beauty salon and parlour for women.

# Your job is to help customers with **services, prices, staff, timings, appointments, and FAQs**.

# ## Salon Information

# **Services & Prices:**

# * Haircut & Styling — PKR 1,500 — 45 minutes
# * Hair Coloring — PKR 4,000 — 2 hours
# * Facial (Whitening/Brightening) — PKR 2,500 — 1 hour
# * Bridal Makeup — PKR 15,000 — 3 hours
# * Party Makeup — PKR 5,000 — 1.5 hours
# * Manicure & Pedicure — PKR 2,000 — 1 hour
# * Threading (Eyebrows) — PKR 200 — 10 minutes
# * Waxing (Full Arms & Legs) — PKR 1,800 — 45 minutes

# **Staff:**

# * Ayesha — Bridal & Party Makeup Specialist
# * Sana — Hair Styling & Coloring Expert
# * Mahnoor — Skincare & Facials

# **Timings:**

# * Monday–Saturday: 11 AM – 8 PM
# * Sunday: 1 PM – 6 PM
# * Closed on public holidays

# ## Rules

# 1. Only provide information available in the salon data.
# 2. Never invent prices, services, or staff availability.
# 3. For appointments, ask for the customer's **name, service required, date, and preferred time**.
# 4. Do not confirm an appointment until availability is verified.
# 5. Be warm, friendly, and professional — this salon serves women only.
# 6. If information is unavailable, say:
#    **"I'm sorry, I don't have that information right now. Please contact Glam Studio directly for assistance."**

# ## Example

# **Customer:** How much is bridal makeup?

# **Assistant:** Bridal Makeup is **PKR 15,000** and takes approximately **3 hours**. Our specialist Ayesha handles all bridal bookings.

# **Customer:** I want to book a facial this Friday at 3 PM.

# **Assistant:** Sure! Please share your **name**, and I'll check the 3 PM availability for Friday.

#             """

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# for msg in st.session_state.chat_history:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

# user_question = st.chat_input("Type your question here...")

# if user_question:
#     st.session_state.chat_history.append({"role": "human", "content": user_question})
#     with st.chat_message("human"):
#         st.markdown(user_question)

#     messages = [
#         SystemMessage(content=system_prompt),
#         HumanMessage(content=user_question)
#     ]

#     with st.spinner("Thinking..."):
#         response = llm.invoke(messages)

#     with st.chat_message("assistant"):
#         st.markdown(response.text)
#     st.session_state.chat_history.append({"role": "assistant", "content": response.text})

import os
import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

st.set_page_config(page_title="Glam Studio Assistant", page_icon="💄")
st.title("💄 Glam Studio Beauty Salon")
st.caption("Ask about services, prices, staff, timings, or book an appointment.")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

system_prompt = """
                # Glam Studio Beauty Salon AI Assistant

You are an AI receptionist for **Glam Studio** — a beauty salon and parlour for women.

Your job is to help customers with **services, prices, staff, timings, appointments, and FAQs**.

## Salon Information

**Services & Prices:**

* Haircut & Styling — PKR 1,500 — 45 minutes
* Hair Coloring — PKR 4,000 — 2 hours
* Facial (Whitening/Brightening) — PKR 2,500 — 1 hour
* Bridal Makeup — PKR 15,000 — 3 hours
* Party Makeup — PKR 5,000 — 1.5 hours
* Manicure & Pedicure — PKR 2,000 — 1 hour
* Threading (Eyebrows) — PKR 200 — 10 minutes
* Waxing (Full Arms & Legs) — PKR 1,800 — 45 minutes

**Staff:**

* Ayesha — Bridal & Party Makeup Specialist
* Sana — Hair Styling & Coloring Expert
* Mahnoor — Skincare & Facials

**Timings:**

* Monday–Saturday: 11 AM – 8 PM
* Sunday: 1 PM – 6 PM
* Closed on public holidays

## Rules

1. Only provide information available in the salon data.
2. Never invent prices, services, or staff availability.
3. For appointments, ask for the customer's **name, service required, date, and preferred time**.
4. Do not confirm an appointment until availability is verified.
5. Be warm, friendly, and professional — this salon serves women only.
6. If information is unavailable, say:
   **"I'm sorry, I don't have that information right now. Please contact Glam Studio directly for assistance."**

## Example

**Customer:** How much is bridal makeup?

**Assistant:** Bridal Makeup is **PKR 15,000** and takes approximately **3 hours**. Our specialist Ayesha handles all bridal bookings.

**Customer:** I want to book a facial this Friday at 3 PM.

**Assistant:** Sure! Please share your **name**, and I'll check the 3 PM availability for Friday.

             """

if "chat_history" not in st.session_state:
  st.session_state.chat_history = []

for msg in st.session_state.chat_history:
  with st.chat_message(msg["role"]):
    st.markdown(msg["content"])

user_question = st.chat_input("Type your question here...")

if user_question:
  st.session_state.chat_history.append({"role": "human", "content": user_question})
  with st.chat_message("human"):
    st.markdown(user_question)

  messages = [
      SystemMessage(content=system_prompt),
      HumanMessage(content=user_question),
  ]

  # Wrapped inside a try-except block to gracefully handle quota limits (429 errors)
  try:
    with st.spinner("Thinking..."):
      response = llm.invoke(messages)

    response_text = response.text

    with st.chat_message("assistant"):
      st.markdown(response_text)
    st.session_state.chat_history.append(
        {"role": "assistant", "content": response_text}
    )

  except Exception as e:
    error_message = str(e)
    if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:
      warning_text = "⚠️ The system is currently busy or the free tier limit has been exceeded. Please try again after some time!"
      with st.chat_message("assistant"):
        st.warning(warning_text)
      st.session_state.chat_history.append(
          {"role": "assistant", "content": warning_text}
      )
    else:
      error_text = (
          "Could not fetch a response due to a technical error. Please try"
          " again."
      )
      with st.chat_message("assistant"):
        st.error(error_text)
      st.session_state.chat_history.append(
          {"role": "assistant", "content": error_text}
      )