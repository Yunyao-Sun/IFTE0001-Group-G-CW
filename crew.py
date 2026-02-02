
import sys
import os
sys.path.append(os.getcwd())
from typing import List
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai import LLM
import crewai.tools
from tools.calculator_tool import CalculatorTool
from tools.RSI_tool import RSITechnicalTool
from tools.MACD_tool import MACDTechnicalTool
from tools.MA_tool import MATechnicalTool



from langchain_openai import ChatOpenAI


from dotenv import load_dotenv
load_dotenv()



@CrewBase
class StockAnalysisCrew:
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    
    my_llm = LLM(
    model="openai/gpt-4o",
    api_key=os.environ["OPENAI_API_KEY"],
    #openai_api_base=os.getenv("OPENAI_API_BASE"),
    temperature=0.2, 
    max_tokens=4096
    )

    @agent
    def technical_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['technical_analyst'],
            verbose=True,
            llm=self.my_llm,
            allow_delegation=False,
            # We inject the powerful new tool here
            tools=[
                MACDTechnicalTool(),
                MATechnicalTool(),
                RSITechnicalTool(),
                CalculatorTool() # Optional: Keep if you want it to check math
            ]
        )
    
    @task
    def technical_analysis(self) -> Task: 
        return Task(
            config=self.tasks_config['technical_analysis'],
            agent=self.technical_agent(),
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the Stock Analysis Crew"""
        return Crew(
            agents=self.agents,  
            tasks=self.tasks, 
            process=Process.sequential,
            verbose=True,
        )



