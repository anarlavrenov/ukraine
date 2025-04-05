from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent
from langchain.agents.agent_toolkits.jira.toolkit import JiraToolkit
from langchain.utilities.jira import JiraAPIWrapper
from agno.agent import Agent
from agno.tools.firecrawl import FirecrawlTools


class OpenAIJiraNoteAgent:
    """
    Represents an agent for converting audio to tasks and creating Jira tickets.

    This class provides functionality to transcribe audio, break down the transcribed text into
    tasks, and generate Jira tickets based on the extracted tasks. It integrates with APIs such
    as OpenAI for transcription and task breakdown and Jira for ticket creation.

    The agent is designed for project management workflows that require seamless transition from
    spoken instructions to actionable Jira tickets.

    The following dependencies are required for this agent:
        pip install atlassian-python-api

    The following environmental variables must be set before using the agent:
        os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
        os.environ["JIRA_API_TOKEN"] = "YOUR_JIRA_API_TOKEN"
        os.environ["JIRA_USERNAME"] = "your_username@gmail.com"
        os.environ["JIRA_INSTANCE_URL"] = "https://project_name.atlassian.net"
        os.environ["JIRA_CLOUD"] = "True"

    Example usage:
        agent = OpenAIJiraNoteAgent()
        transcription = agent.transcribe_audio("audio_2025-03-08_13-26-31.ogg")
        tasks = agent.convert_text_to_tasks(transcription, max_tokens=368)
        agent.create_jira_tickets(tasks, project_key="KAN", language"ukrainian")
    """

    def __init__(
            self
    ):
        pass

    # noinspection PyShadowingNames
    @staticmethod
    def transcribe_audio(path_to_audiofile: str) -> str:
        client = OpenAI()
        audio_file = open(path_to_audiofile, "rb")
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
        return transcription

    @staticmethod
    def convert_text_to_tasks(
            transcribed_text_from_audio: str,
            max_tokens: int = 368
    ) -> str:
        user_content = f"""
              You are a project manager. I want you to break down the message 
              that I will send you into a list of tasks. 
              You should describe the tasks in a way that the team fully understands them. 
              Try to write in detail and clearly, and feel free to add relevant details on your own. 
              The length of the output text should not exceed {max_tokens} tokens.
        """

        # noinspection PyArgumentList
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            max_tokens=max_tokens
        )
        messages = [
            ("system", user_content),
            ("human", transcribed_text_from_audio)
        ]
        ai_msg = llm.invoke(messages)

        return ai_msg.content

    # noinspection PyShadowingNames
    @staticmethod
    def create_jira_tickets(
            tasks: str,
            project_key: str,
            language: str
    ) -> None:
        instruction = f"""
              You are an AI assistant specializing in creating tickets in Jira in project {project_key}. 
              You can create multiple tickets related to a specific project, 
              issue type, title, and description. Do not set priority. 
              Generate descriptions in your own words based on the task descriptions. 
              A ticket can be created within a specific project and issue type. 
              Use the issue type key "TASK".
              The project key and issue type must have the same format: dict.
              Based on the given text, you need to determine 
              how many tickets should be created and open them. 
              The language for ticket titles and descriptions is {language}. 
              Based on this information, I need to open tickets for {tasks}.
        """

        # noinspection PyArgumentList
        llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
        jira = JiraAPIWrapper()
        toolkit = JiraToolkit.from_jira_api_wrapper(jira)
        # noinspection PyShadowingNames
        agent = initialize_agent(
            tools=toolkit.get_tools(),
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        agent.run(instruction)


def parse_web_content(
        link: str,
        prompt: str
) -> str:
    """
    Fetches the content from the given web link based on the supplied prompt
    using an agent with configured tools. The method uses scraping functionality
    without crawling to retrieve the content. The prompt provided is appended
    to the link to form the query for the agent.

    :param link: The URL of the web resource to fetch content from.
    :type link: str
    :param prompt: The instruction or query to guide the content parsing.
    :type prompt: str
    :return: The parsed content extracted from the web link based on the prompt.
    :rtype: str
    """
    parse_agent = Agent(
        tools=[FirecrawlTools(scrape=True, crawl=False)],
        show_tool_calls=False,
        markdown=False
    )
    response = parse_agent.run(prompt + " " + link)
    return response.content
