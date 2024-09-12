from groq import Groq
import json
from duckduckgo_search import DDGS  # Importing DuckDuckGo search library
import time
import os
from rich import print as pprint
from openai import OpenAI
from pydantic import BaseModel, AnyHttpUrl
from typing import List

class FactItemModel(BaseModel):
    fact: str
    is_true:bool
    reason: str
    sources:list[str]


class FactsCollectionModel(BaseModel):
    facts: List[FactItemModel]

class TextProcessor:

    def __init__(self):
        self.promptFile = "prompt5.txt"
        self.groqClient = Groq(api_key=os.environ['GROQ_API_KEY'])

        self.promptContent = ""
        self.transcriptionFile = "transcriptions.txt"
        self.ddgs = DDGS()
        self.llmModel = "llama3-groq-70b-8192-tool-use-preview"
        self.speakers = ""
        self.openAIClient = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        with open(self.promptFile, "r") as fi:
            self.promptContent = fi.read()
    
    def parse_transcript(self):

        file =  open (self.transcriptionFile,"r+")
        lines = "".join(file.readlines()).strip()
        file.truncate(0)

        return lines

    def search_query_tool(self, query_str):
        '''
        Search query using DDGS search
        query: String with queries sperated by comma
        '''

        if type(query_str) != type("string"):
            print(type(query_str))
            pprint("[bold red ]invalid response [/bold red]")
            return "query_str should be string"

        queries = query_str.split(",")[0:-1]
        pprint(f"[bold green] {queries} [/bold green]")
        
        final_result = []

        try:
            for q in queries:

                search_results = self.ddgs.text(q, max_results=5)
                data = {
                    "query":q,
                    "search_results":search_results
                }
                final_result.append(data)
                time.sleep(1)
            pprint("[red] "  + final_result + "[/red]")
            
            return json.dumps(final_result)

        
        except Exception as e:
            pprint("[bold red] Error Occured! [/bold red]")
            print(e)
            return "Search query  tool having problems, please use your own knowledge"
    
    def retrieve_answers_for_questions(self,speaker, questions):

        if isinstance(questions,list):
            final_result = []
            for question in questions:
                search_results = self.ddgs.text(question, max_results=5)
                data = {
                    "question":question,
                    "speaker":speaker,
                    "results":search_results
                }
                final_result.append(data)
                time.sleep(1)
            

            return json.dumps(final_result)
        
        else:
            return "Questions should be a list of questions"
    
    def set_speakers(self,speakers):
        self.speakers = speakers
        

    def feed_to_llm(self,transcript):
        messages = [
            {
                "role": "system",
                "content": self.promptContent
            },
            {
                "role": "user",
                "content": self.speakers  + transcript,
            }
        ]

        tools = [
    

            {
                "type": "function",
                "function": {
                    "name": "retrieve_answers_for_questions",
                    "description": "Retrieve answers for questions using web",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "speaker": {
                                "type": "string",
                                "description": "Current active speaker"
                            },
                            "questions": {
                                "type": "list",
                                "description": "a list of questions"
                            }
                            
                        },
                        "required": ["speaker","questions"],
                    },
                },
            }
        ]

        response = self.groqClient.chat.completions.create(
            model = self.llmModel,
            messages= messages,
            tools=tools,
            tool_choice="required",
            max_tokens=1000,
   
        )
        respnse_message = response.choices[0].message
        tool_calls = respnse_message.tool_calls
        func_mapping = {
            "retrieve_answers_for_questions": self.retrieve_answers_for_questions
        }
        
        if tool_calls:
            print("tool calls")
            total_response = []
            for tool_call in tool_calls:
                func_name = tool_call.function.name
                func_to_call = func_mapping[func_name]
                func_params = json.loads(tool_call.function.arguments)
                print(tool_call.function.arguments)
               
                if func_name == "retrieve_answers_for_questions":
                    response_from_func = func_to_call(questions=func_params["questions"],speaker=func_params['speaker'])
                    total_response.append(response_from_func)
               
            completion = self.openAIClient.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": open(
                        "prompt-final-completetions.txt", "r").read()},
                    {"role": "user", "content": json.dumps(total_response)},
                ],
                response_format=FactsCollectionModel,
            )
            output = completion.choices[0].message

            if output.parsed:
                return output.model_dump_json(indent=2)
            elif output.refusal:
                return "Model Called Failed"

            else:
                print("SOmething")

            return output
           
        print("Tool isnt called and output is", respnse_message.content)
        return respnse_message.content

    def process(self):
        transcript = self.parse_transcript()

        return self.feed_to_llm(transcript=transcript)


