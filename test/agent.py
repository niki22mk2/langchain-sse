from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.agents import ZeroShotAgent, AgentExecutor, Tool, load_tools, initialize_agent

from tools.weather import get_weather
from tools.utils import get_date

prefix = """Answer the following questions as best you can. If you have no specific information, just answer "no information".
You have access to the following tools:
Please observe the input format of each tool. Do not translate the language of the question."""

suffix = """Begin!
Recent Chat history:
{history}

Question: {input}
{agent_scratchpad}"""

def zeroshot(query, history=""):

    weather_forecast_tool = Tool(name="Weather forecast", func=get_weather, description="Useful for answering weather forecasts. Input values must be in Japanese only.")

    llm = ChatOpenAI(temperature=0)

    tools = [
        weather_forecast_tool
    ]

    prompt = ZeroShotAgent.create_prompt(
        tools, 
        prefix=prefix, 
        suffix=suffix, 
        input_variables=["input", "history", "agent_scratchpad"]
    )

    agent = ZeroShotAgent(llm_chain=LLMChain(llm=llm, prompt=prompt), allowed_tools=[tool.name for tool in tools])
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, max_iterations=3, early_stopping_method="generate", request_timeout=60.0)

    agent = initialize_agent(tools, llm, agent="chat-zero-shot-react-description")

    res = agent_executor.run(input=query, history=history)

    return res