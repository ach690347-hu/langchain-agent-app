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
    "marketing.db",
    check_same_thread=False
)

db.execute("""
CREATE TABLE IF NOT EXISTS clients(
    id integer primary key autoincrement,
    name text unique not null,
    service TEXT
)
""")

db.commit()

# ============================================================
# 2. ADD CLIENT
# ============================================================

@tool
def add_client(name: str, service: str):
    """
    Add a new client to the agency database.
    """
    try:
        db.execute("""
        insert into clients(name, service)
        values (?,?)
        """, (name, service))

        db.commit()

        return (
            f"client {name} was added successfully "
            f"with service {service}."
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
    select id, name, service
    from clients
    where name = ?
    """, (name,)).fetchone()

    if client:
        return (
            f"id: {client[0]}, "
            f"name: {client[1]}, "
            f"service: {client[2]}"
        )

    return f"client {name} was not found."

# ============================================================
# 4. SHOW ALL CLIENTS
# ============================================================

@tool
def show_clients():
    """
    Show all clients in the agency database.
    """
    clients = db.execute("""
    select id, name, service
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
            f"service: {client[2]}\n"
        )

    return result

# ============================================================
# 5. UPDATE CLIENT
# ============================================================

@tool
def update_client(name: str, new_service: str):
    """
    Change a client's service.
    """
    result = db.execute("""
    update clients
    set service = ?
    where name = ?
    """, (new_service, name))

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
    Delete a client from the agency database.
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
# 7. SERVICE PRICE
# ============================================================

@tool
def get_service_price(service: str):
    """
    Get the price of a marketing service.
    """
    prices = {
        "social media management": "30000 pkr",
        "seo": "25000 pkr",
        "facebook ads": "20000 pkr",
        "google ads": "22000 pkr",
        "logo design": "15000 pkr",
        "website development": "80000 pkr",
        "video editing": "12000 pkr"
    }

    service = service.lower().strip()

    if service in prices:
        return (
            f"{service.title()} price is "
            f"{prices[service]}."
        )

    return (
        "Sorry, that service is not available. "
        "Available services are: social media management, seo, "
        "facebook ads, google ads, logo design, "
        "website development, video editing."
    )

# ============================================================
# 8. SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a friendly AI receptionist for a digital marketing agency in Pakistan.

You manage clients using a SQLite database.

You can perform these tasks:

1. Add clients.
2. Find clients.
3. Show all clients.
4. Update client services.
5. Delete clients.
6. Give service prices.

IMPORTANT RULES:

- ALWAYS use database tools when changing client information.
- NEVER invent client information.
- NEVER invent service prices.
- Client names must be unique.
- Two clients can have the same service.
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
        get_service_price
    ],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=memory
)

# ============================================================
# 12. CHAT CONFIG
# ============================================================

config = {
    "configurable": {
        "thread_id": "marketing_client_001"
    }
}

print("================================")
print("      AI MARKETING RECEPTIONIST")
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
