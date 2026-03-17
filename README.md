# Local-RAG
A local RAG suitable for running on-device such as with a laptop, no GPU needed, but capable of fitting some RAG layers into GPU acceleration. This entire codebase was vibe coded with Perplexity AI using the Claude Sonnet 4.5 Thinking model under Perplexity Pro.

No subscription or API key is needed for this local RAG.

Available files within this repo include the local RAG Python script, alongside the full Perplexity thread contents with all 80+ prompts plus responses (in both Markdown format and PDF format). "PCS40T" stands for "Perplexity Claude Sonnet 4.0 Thinking" - which was the model version at the time that the original Perplexity thread was started (which transitioned into PCS45T as v4.5 rather than v4.0 before the end of the same thread, and was entirely prompted/generated within October 2025).

## Getting Started

### Assuming an already existing Ollama local installation with at least one LLM available locally
Install micromamba (a C++ optimized version of conda), and run the following two commands, after placing the main Python script into `~/Streamlit/` (or adjust the path within the recommended command accordingly):

`micromamba create -n LocalRAG -c conda-forge streamlit chromadb sentence-transformers langchain-ollama langchain-community pymupdf sqlite tensorflow tf-keras`

`cd ~/Streamlit/; micromamba activate LocalRAG; streamlit run RAGbyPerplexity.py`

The first command only needs to be run once in order to prepare all dependencies for the local RAG. The second command can be re-used in order to launch the local RAG (which will load a Streamlit interface within the default browser). 

### Configuring Ollama Models: Some Recommended LLMs and Embedding Models

This local RAG will detect all models presently available locally through `ollama`, as well as through the Hugging Face command line (currently as `hf` rather than the deprecated `huggingface-cli` version). The current local RAG version looks to `hf` for the list of embedding models available locally, thus `hf` must be used in order to pull embedding models that can be seen by the local RAG. `ollama` embedding models are not presently visible/usable by the local RAG. On the other hand, LLMs presently will only be visible to this local RAG through `ollama` (and not `hf`).

Some recommended models used towards testing biomedical document summarization with this local RAG include:
- `granite4:tiny-h` (as an `ollama` LLM)
- `granite4:micro-h` (as an `ollama` LLM)
- `ibm-granite/granite-embedding-english-r2` (as a `hf` embedding model)
- `ibm-granite/granite-embedding-small-english-r2` (as a `hf` embedding model)
- `pritamdeka/S-PubMedBert-MS-MARCO` (as a `hf` embedding model)
- `sentence-transformers/all-MiniLM-L6-v2` (as a `hf` embedding model)

The first run of this local RAG may pull some of these models automatically. Refer to `ollama` and `hf` documentation to pull any other models desired.
