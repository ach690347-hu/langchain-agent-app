from dotenv import load_dotenv
load_dotenv()

import os
import sqlite3

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

# ============================================================
# 1. DATABASE
# ============================================================

db = sqlite3.connect(
    "realestate.db",
    check_same_thread=False
)

db.execute("""
CREATE TABLE IF NOT EXISTS clients(
    id integer primary key autoincrement,
    name text unique not null,
    requirement TEXT
)
""")

db.commit()

# ============================================================
# 2. ADD CLIENT
# ============================================================

@tool
def add_client(name: str, requirement: str):
    """
    Add a new client to the real estate database.
    """
    try:
        db.execute("""
        insert into clients(name, requirement)
        values (?,?)
        """, (name, requirement))

        db.commit()

        return (
            f"client {name} was added successfully "
            f"with requirement {requirement}."
        )
    except sqlite3.IntegrityError:
        return f"client {name} already exists."

# ============================================================
# 3. FIND CLIENT
# ============================================================

@tool
def find_client(name: str):
    """
    Find a client by name.
    """
    client = db.execute("""
    select id, name, requirement
    from clients
    where name = ?
    """, (name,)).fetchone()

    if client:
        return (
            f"id: {client[0]}, "
            f"name: {client[1]}, "
            f"requirement: {client[2]}"
        )

    return f"client {name} was not found."

# ============================================================
# 4. SHOW ALL CLIENTS
# ============================================================

@tool
def show_clients():
    """
    Show all clients in the real estate database.
    """
    clients = db.execute("""
    select id, name, requirement
    from clients
    order by id
    """).fetchall()

    if not clients:
        return "there are no clients in the database."

    result = ""

    for client in clients:
        result += (
            f"id: {client[0]} "
            f"name: {client[1]} "
            f"requirement: {client[2]}\n"
        )

    return result

# ============================================================
# 5. UPDATE CLIENT
# ============================================================

@tool
def update_client(name: str, new_requirement: str):
    """
    Change a client's requirement.
    """
    result = db.execute("""
    update clients
    set requirement = ?
    where name = ?
    """, (new_requirement, name))

    db.commit()

    if result.rowcount == 0:
        return f"client {name} was not found."

    return f"client {name} was updated successfully."

# ============================================================
# 6. DELETE CLIENT
# ============================================================

@tool
def delete_client(name: str):
    """
    Delete a client from the real estate database.
    """
    result = db.execute("""
    delete from clients
    where name = ?
    """, (name,))

    db.commit()

    if result.rowcount == 0:
        return f"client {name} was not found."

    return f"client {name} was deleted successfully."

# ============================================================
# 7. PROPERTY PRICE
# ============================================================

@tool
def get_property_price(property_type: str):
    """
    Get the price of a property type.
    """
    prices = {
        "dha house": "25000000 pkr",
        "dha plot": "15000000 pkr",
        "gulberg flat": "12000000 pkr",
        "bahria house": "18000000 pkr",
        "model town plot": "16000000 pkr",
        "johar flat": "8500000 pkr"
    }

    property_type = property_type.lower().strip()

    if property_type in prices:
        return (
            f"{property_type.title()} price is "
            f"{prices[property_type]}."
        )

    return (
        "Sorry, that property is not available. "
        "Available properties are: dha house, dha plot, "
        "gulberg flat, bahria house, model town plot, johar flat."
    )

# ============================================================
# 8. SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a friendly AI property assistant for a real estate agency in Pakistan.

You manage clients using a SQLite database.

You can perform these tasks:

1. Add clients.
2. Find clients.
3. Show all clients.
4. Update client requirements.
5. Delete clients.
6. Give property prices.

IMPORTANT RULES:

- ALWAYS use database tools when changing client information.
- NEVER invent client information.
- NEVER invent property prices.
- Client names must be unique.
- Two clients can have the same requirement.
- When adding a new client, ALWAYS ask for their full requirement first.
  Ask one by one:
  "Aap kis jagah ghar lena chahte hain?"
  "Kis type ki property chahiye — house, plot ya flat?"
  "Buy, sell ya rent karna hai?"
  "Budget kya hai?"
  "Kab tak chahiye (timeline)?"
- Only call add_client after the client has given their requirement.
  Save the full requirement (area, property type, buy/sell/rent,
  budget, timeline) in the requirement field.
- If the user asks to add multiple clients,
  call add_client for each client.
- If the user asks to delete multiple clients,
  call delete_client for each client.
- If the user asks to update multiple clients,
  call update_client for each client.
- If the user asks to add and delete clients in the
  same request, perform all required operations.
- Understand natural language.

Do NOT just tell the user that you performed the changes.
Actually call the tools.

After completing the operations, give a short friendly
confirmation.

Be concise.
"""


# ============================================================
# 9. GEMINI
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# ============================================================
# 10. MEMORY
# ============================================================

memory = MemorySaver()

# ============================================================
# 11. CREATE AGENT
# ============================================================

agent = create_agent(
    model=llm,
    tools=[
        add_client,
        find_client,
        show_clients,
        update_client,
        delete_client,
        get_property_price
    ],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=memory
)

# ============================================================
# 12. CHAT CONFIG
# ============================================================

config = {
    "configurable": {
        "thread_id": "realestate_client_001"
    }
}

print("================================")
print("      AI REAL ESTATE ASSISTANT")
print("================================")
print("Type 'exit' to stop.")

while True:
    user_input = input("\nClient: ")

    if user_input.lower().strip() == "exit":
        print("\nGoodbye!")
        break

    if not user_input.strip():
        continue

    try:
        response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_input
                    }
                ]
            },
            config=config
        )

        # Get the last AI message
        answer = response["messages"][-1].text

        print("\nAI:", answer)

    except Exception as e:
        print("\nError:", e)
