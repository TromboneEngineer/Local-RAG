# Local-RAG
A local RAG suitable for running on-device such as with a laptop, no GPU needed, but capable of fitting some RAG layers into GPU acceleration. This entire codebase was vibe coded with Perplexity AI using the Claude Sonnet 4.5 Thinking model under Perplexity Pro.

## Getting Started

### Assuming an already existing Ollama local installation with at least one LLM available locally
Install micromamba (a C++ optimized version of conda), and run the following two commands, after placing the main Python script into `~/Streamlit/` (or adjust the path within the recommended command accordingly):

`micromamba create -n LocalRAG -c conda-forge streamlit chromadb sentence-transformers langchain-ollama langchain-community pymupdf sqlite tensorflow tf-keras`

`cd ~/Streamlit/; micromamba activate LocalRAG; streamlit run RAGbyPerplexity.py`
