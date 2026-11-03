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
    "restaurant.db",
    check_same_thread=False
)

db.execute("""
CREATE TABLE IF NOT EXISTS customers(
    id integer primary key autoincrement,
    name text unique not null,
    favorite_dish TEXT
)
""")

db.commit()

# ============================================================
# 2. ADD CUSTOMER
# ============================================================

@tool
def add_customer(name: str, favorite_dish: str):
    """
    Add a new customer to the restaurant database.
    """
    try:
        db.execute("""
        insert into customers(name, favorite_dish)
        values (?,?)
        """, (name, favorite_dish))

        db.commit()

        return (
            f"customer {name} was added successfully "
            f"with favorite dish {favorite_dish}."
        )
    except sqlite3.IntegrityError:
        return f"customer {name} already exists."

# ============================================================
# 3. FIND CUSTOMER
# ============================================================

@tool
def find_customer(name: str):
    """
    Find a customer by name.
    """
    customer = db.execute("""
    select id, name, favorite_dish
    from customers
    where name = ?
    """, (name,)).fetchone()

    if customer:
        return (
            f"id: {customer[0]}, "
            f"name: {customer[1]}, "
            f"favorite dish: {customer[2]}"
        )

    return f"customer {name} was not found."

# ============================================================
# 4. SHOW ALL CUSTOMERS
# ============================================================

@tool
def show_customers():
    """
    Show all customers in the restaurant database.
    """
    customers = db.execute("""
    select id, name, favorite_dish
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
            f"favorite dish: {customer[2]}\n"
        )

    return result

# ============================================================
# 5. UPDATE CUSTOMER
# ============================================================

@tool
def update_customer(name: str, new_dish: str):
    """
    Change a customer's favorite dish.
    """
    result = db.execute("""
    update customers
    set favorite_dish = ?
    where name = ?
    """, (new_dish, name))

    db.commit()

    if result.rowcount == 0:
        return f"customer {name} was not found."

    return f"customer {name} was updated successfully."

# ============================================================
# 6. DELETE CUSTOMER
# ============================================================

@tool
def delete_customer(name: str):
    """
    Delete a customer from the restaurant database.
    """
    result = db.execute("""
    delete from customers
    where name = ?
    """, (name,))

    db.commit()

    if result.rowcount == 0:
        return f"customer {name} was not found."

    return f"customer {name} was deleted successfully."

# ============================================================
# 7. DISH PRICE
# ============================================================

@tool
def get_dish_price(dish: str):
    """
    Get the price of a dish.
    """
    prices = {
        "chicken biryani": "450 pkr",
        "chicken karahi": "1400 pkr",
        "zinger burger": "550 pkr",
        "chicken pizza": "1100 pkr",
        "bbq platter": "1800 pkr",
        "gulab jamun": "300 pkr"
    }

    dish = dish.lower().strip()

    if dish in prices:
        return (
            f"{dish.title()} price is "
            f"{prices[dish]}."
        )

    return (
        "Sorry, that dish is not available. "
        "Available dishes are: chicken biryani, chicken karahi, "
        "zinger burger, chicken pizza, bbq platter, gulab jamun."
    )

# ============================================================
# 8. SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a friendly AI receptionist for a restaurant in Pakistan.

You manage customers using a SQLite database.

You can perform these tasks:

1. Add customers.
2. Find customers.
3. Show all customers.
4. Update customer favorite dishes.
5. Delete customers.
6. Give dish prices.

IMPORTANT RULES:

- ALWAYS use database tools when changing customer information.
- NEVER invent customer information.
- NEVER invent dish prices.
- Customer names must be unique.
- Two customers can have the same favorite dish.
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
        add_customer,
        find_customer,
        show_customers,
        update_customer,
        delete_customer,
        get_dish_price
    ],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=memory
)

# ============================================================
# 12. CHAT CONFIG
# ============================================================

config = {
    "configurable": {
        "thread_id": "restaurant_customer_001"
    }
}

print("================================")
print("      AI RESTAURANT RECEPTIONIST")
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

        # Get the last AI message
        answer = response["messages"][-1].text

        print("\nAI:", answer)

    except Exception as e:
        print("\nError:", e)
