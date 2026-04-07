# Ukraine

Ukraine is a deep learning toolkit that includes research models, approaches and utils.

## Installation

```bash
pip install -U ukraine[langchain_llama]
```

```python
from ukraine.agents.rag import PDFLlamaRAGAgent

agent = PDFLlamaRAGAgent(
    file_path="PATH_TO_PDF",
    system_prompt="""Provide answers based on the document."{context}"""
)
result = agent.chat("What is this document about?")
print(result["answer"])
```
[View this example in the cookbook](./cookbook/rag_cookbook.ipynb)

### 🧠 Introducing Complex AI Agents

A new file has been added:  `ukraine/agents/complex.py`

Includes the `OpenAIJiraNoteAgent` class for automated conversion of audio file that contains meeting recording into Jira tasks using OpenAI + LangChain integrations.

**Capabilities:**
- Transcribes audio to text using `whisper-1`
- Breaks down transcribed instructions into actionable tasks
- Creates structured Jira tickets via LangChain `JiraToolkit`

---

## Installation

```bash
pip install -U ukraine[jira]
```

---

## Example usage

```python
from ukraine.agents.complex import OpenAIJiraNoteAgent

agent = OpenAIJiraNoteAgent()
transcription = agent.transcribe_audio("audio.ogg")
tasks = agent.convert_text_to_tasks(transcription, max_tokens=368)
agent.create_jira_tickets(tasks, project_key="KAN", language="ukrainian")
```

---

## Environment variables required

```bash
OPENAI_API_KEY=<your_openai_key>
JIRA_API_TOKEN=<your_jira_token>
JIRA_USERNAME=<your_email>
JIRA_INSTANCE_URL=https://yourproject.atlassian.net
JIRA_CLOUD=True
```
