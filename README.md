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
