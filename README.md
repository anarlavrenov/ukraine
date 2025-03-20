# Ukraine

Ukraine is a deep learning toolkit that includes transformer models, tokenizers, and masking utilities.

## Installation

```bash
# !pip install -U ukraine[langchain_llama]
```

```python
from ukraine.agents.rag import PDFLlamaRAGAgent

agent = PDFLlamaRAGAgent(
    file_path="PATH_TO_PDF",
    system_prompt="""Provide answers based on the document."{context}""""
)
result = agent.chat("What is this document about?")
print(result["answer"])
```
