from crewai import Agent, Task, Crew
from crewai_tools import SerpApiGoogleSearchTool
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the search tool
search_tool = SerpApiGoogleSearchTool()

# Define the agent with search capabilities
research_agent = Agent(
    role="AI researcher",
    goal="Search and summarize information accurately",
    backstory="Expert AI agent that can search online and generate summaries.",
    verbose=True,
    tools=[search_tool]
)

# Task for the agent - hardcoded for testing
task = Task(
    description="Provide a summary about: Python programming language",
    agent=research_agent,
    expected_output="A clean, comprehensive summary."
)

# Crew executor
crew = Crew(
    agents=[research_agent],
    tasks=[task]
)

result = crew.kickoff()
print("\n" + "="*50)
print("RESULT:")
print("="*50)
print(result)
