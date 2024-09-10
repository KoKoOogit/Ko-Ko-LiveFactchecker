from groq import Groq
import json
from duckduckgo_search import DDGS  # Importing DuckDuckGo search library
import time
import os
from rich import print as pprint
class TextProcessor:

    def __init__(self):
        self.promptFile = "prompt4.txt"
        self.groqClient = Groq(api_key=os.environ['GROQ_API_KEY'])
        self.promptContent = ""
        self.transcriptionFile = "transcriptions.txt"
        self.ddgs = DDGS()
        self.llmModel = "mixtral-8x7b-32768"

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

    def feed_to_llm(self,transcript):
        messages = [
            {
                "role": "system",
                "content": self.promptContent
            },
            {
                "role": "user",
                "content": transcript,
            }
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_query_tool",
                    "description": "Access Relevant source to check facts and statements",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query_str": {
                                "type": "string",
                                "description": " a string of comma-separated search queries."
                            }
                        },
                        "required": ["query_str"],
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
            "search_query_tool":self.search_query_tool
        }
        if tool_calls:
            for tool_call in tool_calls:
                func_name = tool_call.function.name
                func_to_call = func_mapping[func_name]
                func_params = json.loads(tool_call.function.arguments)
               

                if func_name == "search_query_tool":
                    response_from_func = func_to_call(query_str=func_params)
                
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": func_name,
                    "content": response_from_func
                })

            final_response = self.groqClient.chat.completions.create(
                        model = self.llmModel,
                        messages=messages,
                    )
            
            return final_response.choices[0].message.content
        
        return response.choices[0].message.content

    def process(self):
        transcript = self.parse_transcript()

        return self.feed_to_llm(transcript=transcript)
