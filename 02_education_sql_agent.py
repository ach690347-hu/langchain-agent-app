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
    "education.db",
    check_same_thread=False
)

db.execute("""
CREATE TABLE IF NOT EXISTS students(
    id integer primary key autoincrement,
    name text unique not null,
    course TEXT
)
""")

db.commit()

# ============================================================
# 2. ADD STUDENT
# ============================================================

@tool
def add_student(name: str, course: str):
    """
    Add a new student to the institute database.
    """
    try:
        db.execute("""
        insert into students(name, course)
        values (?,?)
        """, (name, course))

        db.commit()

        return (
            f"student {name} was added successfully "
            f"with course {course}."
        )
    except sqlite3.IntegrityError:
        return f"student {name} already exists."

# ============================================================
# 3. FIND STUDENT
# ============================================================

@tool
def find_student(name: str):
    """
    Find a student by name.
    """
    student = db.execute("""
    select id, name, course
    from students
    where name = ?
    """, (name,)).fetchone()

    if student:
        return (
            f"id: {student[0]}, "
            f"name: {student[1]}, "
            f"course: {student[2]}"
        )

    return f"student {name} was not found."

# ============================================================
# 4. SHOW ALL STUDENTS
# ============================================================

@tool
def show_students():
    """
    Show all students in the institute database.
    """
    students = db.execute("""
    select id, name, course
    from students
    order by id
    """).fetchall()

    if not students:
        return "there are no students in the database."

    result = ""

    for student in students:
        result += (
            f"id: {student[0]} "
            f"name: {student[1]} "
            f"course: {student[2]}\n"
        )

    return result

# ============================================================
# 5. UPDATE STUDENT
# ============================================================

@tool
def update_student(name: str, new_course: str):
    """
    Change a student's course.
    """
    result = db.execute("""
    update students
    set course = ?
    where name = ?
    """, (new_course, name))

    db.commit()

    if result.rowcount == 0:
        return f"student {name} was not found."

    return f"student {name} was updated successfully."

# ============================================================
# 6. DELETE STUDENT
# ============================================================

@tool
def delete_student(name: str):
    """
    Delete a student from the institute database.
    """
    result = db.execute("""
    delete from students
    where name = ?
    """, (name,))

    db.commit()

    if result.rowcount == 0:
        return f"student {name} was not found."

    return f"student {name} was deleted successfully."

# ============================================================
# 7. COURSE FEE
# ============================================================

@tool
def get_course_fee(course: str):
    """
    Get the fee of a course.
    """
    fees = {
        "agentic ai": "40000 pkr",
        "web development": "40000 pkr",
        "social media marketing": "40000 pkr",
        "graphic designing": "25000 pkr",
        "english language": "15000 pkr",
        "spoken english": "12000 pkr"
    }

    course = course.lower().strip()

    if course in fees:
        return (
            f"{course.title()} course fee is "
            f"{fees[course]}."
        )

    return (
        "Sorry, that course is not available. "
        "Available courses are: agentic ai, web development, "
        "social media marketing, graphic designing, "
        "english language, spoken english."
    )

# ============================================================
# 8. SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a friendly AI receptionist for an educational institute in Pakistan.

You manage students using a SQLite database.

You can perform these tasks:

1. Add students.
2. Find students.
3. Show all students.
4. Update student courses.
5. Delete students.
6. Give course fees.

IMPORTANT RULES:

- ALWAYS use database tools when changing student information.
- NEVER invent student information.
- NEVER invent course fees.
- Student names must be unique.
- Two students can have the same course.
- If the user asks to add multiple students,
  call add_student for each student.
- If the user asks to delete multiple students,
  call delete_student for each student.
- If the user asks to update multiple students,
  call update_student for each student.
- If the user asks to add and delete students in the
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
        add_student,
        find_student,
        show_students,
        update_student,
        delete_student,
        get_course_fee
    ],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=memory
)

# ============================================================
# 12. CHAT CONFIG
# ============================================================

config = {
    "configurable": {
        "thread_id": "education_student_001"
    }
}

print("================================")
print("      AI EDUCATION RECEPTIONIST")
print("================================")
print("Type 'exit' to stop.")

while True:
    user_input = input("\nStudent: ")

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
