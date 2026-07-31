# FinSight-Ai

Financial PDF Agentic RAG in Python. Index up to four 10-K style PDFs locally, ask grounded questions with citations, and run arithmetic through a sanitized calculator tool.

## Features

- PDF ingestion with table-aware extraction and local Chroma persistence
- LangGraph agent with document search and financial calculator tools
- Streamlit UI with background multi-PDF indexing (4-file cap) and chat
- Free-tier Gemini embeddings and `gemini-2.5-flash-lite`

## Setup

1. Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and set your Gemini API key:

```env
GEMINI_API_KEY=your_key_here
```

3. Run the app:

```bash
streamlit run app.py
```

## Project layout

```
FinSight-Ai/
├── app.py
├── requirements.txt
├── .env.example
├── .streamlit/
├── scripts/
├── src/
│   ├── config.py
│   ├── ingestion.py
│   ├── tools.py
│   ├── agent.py
│   ├── prompts.py
│   └── index_jobs.py
├── data/
└── chroma_db/
```

## Notes

- Indexed chunks persist under `./chroma_db`
- Uploaded source files are stored under `./data`
- Do not commit `.env` or local vector-store data
