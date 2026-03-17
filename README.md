# Local-RAG
A local RAG suitable for running on-device such as with a laptop, no GPU needed, but capable of fitting some RAG layers into GPU acceleration. This entire codebase was vibe coded with Perplexity AI using the Claude Sonnet 4.5 Thinking model under Perplexity Pro.

## Getting Started
Install micromamba (a C++ optimized version of conda), and run the following two commands:

`micromamba create -n LocalRAG -c conda-forge streamlit chromadb sentence-transformers langchain-ollama langchain-community pymupdf sqlite tensorflow tf-keras`

`cd ~/Streamlit/; micromamba activate LocalRAG; streamlit run RAGbyPerplexity.py`
