from dotenv import load_dotenv
load_dotenv()

import os
import sqlite3

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'"
# 1.  DATABASE
# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

db = sqlite3.connect(
    "barber.db",
    check_same_thread=False
    )


db.execute("""
CREATE TABLE IF NOT EXISTS customers(
    id integer primary key autoincrement,
    name text unique not null,
    favorite_service  TEXT
           )
""")

db.commit()

# """"""""""""""""""""""""""""""""""""""""""""""""""""""
# 2. ADD CUSTOMER
# """""""""""""""""""""""""""""""""""""""""""""""""""""""

@tool
def add_customer(name: str,favorite_service: str):
    """
    Add a new customer to the barber shop database.
    """
    try:

        db.execute("""
        insert into customers(name, favorite_service)
        values (?,?)
        """,(name, favorite_service))

        db.commit()

        return(
            f"customer {name} was added successfully "
            f"with favorite service {favorite_service}."
        )
    except sqlite3.IntegrityError:
        return f"customer {name} already exists."


# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# 3,    FIND_CUSTOMER
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

@tool
def find_customer(name: str):
    """
    FIND a customer by name.
    """
    customer = db.execute("""
 select id, name, favorite_service
 from customers
 where name = ?
 """, (name,)).fetchone()

    if customer:

        return(
            f"id: {customer[0]}, "
            f"name: {customer[1]}, "
            f"favorite service: {customer[2]}"
        )

    return f"customer {name} was not found."




# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# 4.    SHOW ALL CUSTOMERS
# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
@tool
def show_customers():
    """
    Show all customers in the barber shop database.
    """
    customers = db.execute("""
    select id, name, favorite_service
    from customers
    order by id
    """).fetchall()

    if not customers:
        return "there are no customers in the database."

    result = ""

    for customer in customers:
        result += (
            f"id: {customer[0]} "
            f"name: {customer[1]} "
            f"favorite service: {customer[2]}\n"
        )

    return result

# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# 5.  update add_customer
# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

@tool
def update_customer(name: str, new_service: str):
    """
    Change a customer's favorite service.
    """
    result = db.execute("""
    update customers
    set favorite_service = ?
    where name = ?
    """, (new_service, name))

    db.commit()

    if result.rowcount == 0:
        return f"customer {name} was not found."

    return f"customer {name} was updated successfully."


# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# 6.  DELETE CUSTOMER
# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

@tool
def delete_customer(name: str):
    """
    Delete a customer from the barber shop database.
    """
    result = db.execute("""
    delete from customers
    where name = ?
    """, (name,))

    db.commit()

    if result.rowcount == 0:
        return f"customer {name} was not found."

    return f"customer {name} was deleted successfully."


#==========================================================
# 7.  SERVICE PRICE
# =========================================================
@tool
def get_service_price(service: str):
    """
    Get the price of a barber shop service.
    """
    prices = {
        "haircut": "3500 pkr",
        "fade": "3400 pkr",
        "beard": "2000 pkr",
        "haircut and beard": "5000 pkr",
        "haircut, beard and fade": "8999 pkr"
    }

    service = service.lower().strip()

    if service in prices:
        return (
            f"{service.title()} costs "
            f"{prices[service]}."
        )

    return (
        "Sorry, that service is not available. "
        "Available services are: haircut, fade, beard, "
        "haircut and beard, haircut, beard and fade."
    )


# =========================================================
# 8. system_prompt
# ==========================================================

SYSTEM_PROMPT = """
You are a friendly AI receptionist for a USA barber shop.

You manage customers using a SQLite database.

You can perform these tasks:

1. Add customers.
2. Find customers.
3. Show all customers.
4. Update customer favorite services.
5. Delete customers.
6. Give service prices.

IMPORTANT RULES:

- ALWAYS use database tools when changing customer information.
- NEVER invent customer information.
- NEVER invent service prices.
- Customer names must be unique.
- Two customers can have the same favorite service.
- If the user asks to add multiple customers,
  call add_customer for each customer.
- If the user asks to delete multiple customers,
  call delete_customer for each customer.
- If the user asks to update multiple customers,
  call update_customer for each customer.
- If the user asks to add and delete customers in the
  same request, perform all required operations.
- Understand natural language.

Do NOT just tell the user that you performed the changes.
Actually call the tools.

After completing the operations, give a short friendly
confirmation.

Be concise.
"""

# =============================================================
# 9. GEMINI
# =============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


# ========================================================
# 10.Memory
# ========================================================

memory = MemorySaver()

# ============================================================
# 11. CREATE AGENT
# ============================================================
agent = create_agent(
    model=llm,
    tools=[
        add_customer,
        find_customer,
        show_customers,
        update_customer,
        delete_customer,
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
        "thread_id": "barber_customer_001"
    }
}

print("================================")
print("      AI BARBER RECEPTIONIST")
print("================================")
print("Type 'exit' to stop.")

while True:
    user_input = input("\nCustomer: ")


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

    
        answer = response["messages"][-1].text

        print("\nAI:", answer)

    except Exception as e:
        print("\nError:", e)
