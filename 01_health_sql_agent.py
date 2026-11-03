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
    "healthcare.db",
    check_same_thread=False
)

db.execute("""
CREATE TABLE IF NOT EXISTS patients(
    id integer primary key autoincrement,
    name text unique not null,
    doctor TEXT
)
""")

db.commit()

# ============================================================
# 2. ADD PATIENT
# ============================================================

@tool
def add_patient(name: str, doctor: str):
    """
    Add a new patient to the clinic database.
    """
    try:
        db.execute("""
        insert into patients(name, doctor)
        values (?,?)
        """, (name, doctor))

        db.commit()

        return (
            f"patient {name} was added successfully "
            f"with doctor {doctor}."
        )
    except sqlite3.IntegrityError:
        return f"patient {name} already exists."

# ============================================================
# 3. FIND PATIENT
# ============================================================

@tool
def find_patient(name: str):
    """
    Find a patient by name.
    """
    patient = db.execute("""
    select id, name, doctor
    from patients
    where name = ?
    """, (name,)).fetchone()

    if patient:
        return (
            f"id: {patient[0]}, "
            f"name: {patient[1]}, "
            f"doctor: {patient[2]}"
        )

    return f"patient {name} was not found."

# ============================================================
# 4. SHOW ALL PATIENTS
# ============================================================

@tool
def show_patients():
    """
    Show all patients in the clinic database.
    """
    patients = db.execute("""
    select id, name, doctor
    from patients
    order by id
    """).fetchall()

    if not patients:
        return "there are no patients in the database."

    result = ""

    for patient in patients:
        result += (
            f"id: {patient[0]} "
            f"name: {patient[1]} "
            f"doctor: {patient[2]}\n"
        )

    return result

# ============================================================
# 5. UPDATE PATIENT
# ============================================================

@tool
def update_patient(name: str, new_doctor: str):
    """
    Change a patient's doctor.
    """
    result = db.execute("""
    update patients
    set doctor = ?
    where name = ?
    """, (new_doctor, name))

    db.commit()

    if result.rowcount == 0:
        return f"patient {name} was not found."

    return f"patient {name} was updated successfully."

# ============================================================
# 6. DELETE PATIENT
# ============================================================

@tool
def delete_patient(name: str):
    """
    Delete a patient from the clinic database.
    """
    result = db.execute("""
    delete from patients
    where name = ?
    """, (name,))

    db.commit()

    if result.rowcount == 0:
        return f"patient {name} was not found."

    return f"patient {name} was deleted successfully."

# ============================================================
# 7. CONSULTATION FEE
# ============================================================

@tool
def get_consultation_fee(doctor: str):
    """
    Get the consultation fee of a doctor.
    """
    fees = {
        "general physician": "1500 pkr",
        "dentist": "2500 pkr",
        "cardiologist": "3500 pkr",
        "dermatologist": "2000 pkr",
        "pediatrician": "1800 pkr",
        "eye specialist": "2200 pkr"
    }

    doctor = doctor.lower().strip()

    if doctor in fees:
        return (
            f"{doctor.title()} consultation fee is "
            f"{fees[doctor]}."
        )

    return (
        "Sorry, that doctor is not available. "
        "Available doctors are: general physician, dentist, "
        "cardiologist, dermatologist, pediatrician, eye specialist."
    )

# ============================================================
# 8. SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a friendly AI receptionist for a clinic in Pakistan.

You manage patients using a SQLite database.

You can perform these tasks:

1. Add patients.
2. Find patients.
3. Show all patients.
4. Update patient doctors.
5. Delete patients.
6. Give consultation fees.

IMPORTANT RULES:

- ALWAYS use database tools when changing patient information.
- NEVER invent patient information.
- NEVER invent doctor fees.
- Patient names must be unique.
- Two patients can have the same doctor.
- If the user asks to add multiple patients,
  call add_patient for each patient.
- If the user asks to delete multiple patients,
  call delete_patient for each patient.
- If the user asks to update multiple patients,
  call update_patient for each patient.
- If the user asks to add and delete patients in the
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
        add_patient,
        find_patient,
        show_patients,
        update_patient,
        delete_patient,
        get_consultation_fee
    ],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=memory
)

# ============================================================
# 12. CHAT CONFIG
# ============================================================

config = {
    "configurable": {
        "thread_id": "healthcare_patient_001"
    }
}

print("================================")
print("      AI HEALTHCARE RECEPTIONIST")
print("================================")
print("Type 'exit' to stop.")

while True:
    user_input = input("\nPatient: ")

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
