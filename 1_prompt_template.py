# import os
# from dotenv import load_dotenv

# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import PromptTemplate


# load_dotenv()


# llm = ChatGoogleGenerativeAI(
#     model="gemini-3.6-flash",
#     temperature=0,
#     google_api_key=os.getenv("GOOGLE_API_KEY")
# )


# template_string = """

# student_detail:
# name: {student_name}
# matric marks: {matric_marks}
# inter marks: {inter_marks}
# Desired program: {desired_program}

# Based on the student's academic record above, suggest which universities in Lahore (such as FAST, PUCIT, ITU, COMSATS, UET).
# """

# admission_template = PromptTemplate(
#     input_variables=["student_name", "matric_marks", "inter_marks", "desired_program"],
#     template=template_string
# )

# name = input("Enter Student Name: ")
# matric = input("Enter Matric Marks ( 1020/1100): ")
# inter = input("Enter Inter Marks ( 950/1100): ")
# program = input("Enter Desired Program ( BS CS / AI / Software Engineering): ")


# final_prompt = admission_template.format(
#     student_name=name,
#     matric_marks=matric,
#     inter_marks=inter,
#     desired_program=program
# )

# response = llm.invoke(final_prompt)

# print(response.text)
# import os
# from dotenv import load_dotenv

# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import PromptTemplate
# load_dotenv()



# llm = ChatGoogleGenerativeAI(
#     model="gemini-3.6-flash",
#     temperature=0,
#     google_api_key=os.getenv("GOOGLE_API_KEY")
# )

# tempalte_string = "Give me 3 tourist place to visit in {city_name}"
# city_template = PromptTemplate ( 
#     input_varibales = ["city_name"],
#     template=tempalte_string
#     )
# city = input("enter your city")
# final_prompt = city_template.format(city_name=city)

# response = llm.invoke(final_prompt 

# )


# print(response.text)

# import os
# from dotenv import load_dotenv

# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import PromptTemplate
# load_dotenv()



# llm = ChatGoogleGenerativeAI(
#     model="gemini-3.6-flash",
#     temperature=0,
#     google_api_key=os.getenv("GOOGLE_API_KEY")
# )

# template_string = "Give me a recipe for {dish_name} which belongs to {cuisine_type} cuisine"
# dish_template = PromptTemplate (
#     input_variables = ["dish_name", "cuisine_type"],
#     template=template_string)
# dish = input("enter your dish")
# cuisine = input("enter your cuisine")
# Final_prompt = dish_template.format(dish_name=dish,cuisine_type=cuisine)



# response = llm.invoke(Final_prompt)


# print(response.text)

import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
load_dotenv()



llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
template_string = "write a formal leave applicaton for {name} who need {days} days off due to {reason}"
name_template = PromptTemplate (
    input_variables = ["name","days","reason"],
    template = template_string)
name = input("enter your name")
days = input ("enter your days")
reason = input("enter your reason")
Final_prompt = name_template.format(name = name, days=days, reason=reason)




response = llm.invoke(Final_prompt)


print(response.text)