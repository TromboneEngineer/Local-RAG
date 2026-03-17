# Local-RAG
A local RAG suitable for running on-device such as with a laptop, no GPU needed, but capable of fitting some RAG layers into GPU acceleration. This entire codebase was vibe coded with Perplexity AI using the Claude Sonnet 4.5 Thinking model under Perplexity Pro. No subscription or API key is needed for this local RAG. Available files within this repo include the local RAG Python script, alongside the full Perplexity thread contents with all 80+ prompts plus responses (in both Markdown format and PDF format). "PCS40T" stands for "Perplexity Claude Sonnet 4.0 Thinking" - which was the model version at the time that the original Perplexity thread was started (which transitioned into PCS45T as v4.5 rather than v4.0 before the end of the same thread, and was entirely prompted/generated within October 2025).

## Getting Started

### Assuming an already existing Ollama local installation with at least one LLM available locally
Install micromamba (a C++ optimized version of conda), and run the following two commands, after placing the main Python script into `~/Streamlit/` (or adjust the path within the recommended command accordingly):

`micromamba create -n LocalRAG -c conda-forge streamlit chromadb sentence-transformers langchain-ollama langchain-community pymupdf sqlite tensorflow tf-keras`

`cd ~/Streamlit/; micromamba activate LocalRAG; streamlit run RAGbyPerplexity.py`
