from groq import Groq
import json
from duckduckgo_search import DDGS  # Importing DuckDuckGo search library
import time
import os
from dotenv import load_dotenv

# load_dotenv()
client = Groq(api_key=os.environ['GROQ_API_KEY'])
MODEL = 'llama3-8b-8192'

with open("prompt2.txt","r") as fi:
    prompt = fi.read()

def calculate(expression):
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return json.dumps({"result": result})
    except:
        return json.dumps({"error": "Invalid expression"})


def fact_check(questions_str):
    questions = questions_str.split(",")[0:-1]
    final_results = []
    ddgs = DDGS()
    try:
        for q in questions:
            result = ddgs.text(q,max_results=5)
            data = {
                "question":q,
                "search_results":result
            }
            final_results.append(data)
            time.sleep(1)
        
        return json.dumps(final_results)

    except Exception as e:
        print(e)
        return "Error Cannot Found"

def run_conversation(user_prompt):
    messages = [
        {
            "role": "system",
            "content": prompt
        },
        {
            "role": "user",
            "content": user_prompt,
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Evaluate a mathematical expression",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The mathematical expression to evaluate",
                        }
                    },
                    "required": ["expression"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fact_check",
                "description": "Access Relevant source to check facts and statements",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "questions_str":{
                            "type":"string",
                            "description": "Search queue sperated by comma"
                        }
                    },
                    "required": ["questions_str"],
                },
            },
        }
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=4096
    )

    response_message = response.choices[0].message
    print(response_message)
    tool_calls = response_message.tool_calls
    if tool_calls:
        print("Tool Calls")
        available_functions = {
            "calculate": calculate,
            "fact_check": fact_check
        }
        messages.append(response_message)
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)

            # Dynamically call the appropriate function
            if function_name == "calculate":
                function_response = function_to_call(
                    expression=function_args.get("expression"))
            elif function_name == "fact_check":
                function_response = function_to_call(
                    questions_str=function_args.get("questions_str"))
                print(function_response)

            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )
        second_response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        return second_response.choices[0].message.content

def parse_contents(filepath):

        # Step 1: Read the content of the text file
    file_path = filepath  # Replace with your file path

    with open(file_path, "r") as file:
        lines = file.readlines()

    # Step 2: Check if there are lines to process
    if len(lines) > 1:
        # Step 3: Store all but the last line in a variable
        parsed_content_variable = "".join(lines[:-1]).strip()

        # Step 4: Keep only the last line in the file
        with open(file_path, "w") as file:
            file.write(lines[-1].strip())

        # Output the results
        print("Parsed content stored in variable:")
        print(parsed_content_variable)
    else:
        print("The file does not contain multiple lines or is empty.")

# Example usage with multiple facts

simple_prompt = '''
Speaker 1:  My running mate is Obama
Spekar 2: The earth is rectangular

Speaker 1 Kamala Harris
Speaker 2 Unknown
'''
a = run_conversation(simple_prompt)
print(a)
