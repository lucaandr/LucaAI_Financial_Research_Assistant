# LucaAI — Financial Research Assistant

LucaAI is a Streamlit web app for financial research. Ask questions about companies and stocks, search uploaded financial reports, and review market data and recent news.

## Features

- An AI agent that uses financial research tools to answer questions.
- Market data and technical indicators for stock tickers.
- Analyst ratings and price targets.
- Recent insider transaction information.
- Uploading and indexing financial PDF reports, including 10-K and 10-Q filings.
- Semantic search across indexed documents.
- Recent financial news search.
- Evidence coverage checks for numeric claims.
- Persistent conversation history and document records.
- Docker Compose support.

## Technologies

Python, Streamlit, LangChain, Groq, ChromaDB, Sentence Transformers, yfinance, and Firecrawl.

## Requirements

- Python 3.11 or Docker with Docker Compose.
- A Groq API key.
- A Firecrawl API key is optional. Without it, the app falls back to DuckDuckGo for web search.

## Local setup

1. Clone the repository and navigate to the project directory.
2. Create a virtual environment and install the dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

   On Windows PowerShell, activate the environment with:

   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

3. Create a `.env` file in the project root:

   ```env
   GROQ_API_KEY=your_groq_api_key
   FIRECRAWL_API_KEY=your_firecrawl_api_key
   ```

   `FIRECRAWL_API_KEY` is optional.

4. Start the app:

   ```bash
   streamlit run app.py
   ```

## Run with Docker

Create the `.env` file as described above, then run:

```bash
docker compose up --build
```

Open the app at [http://localhost:8501](http://localhost:8501).

## Working with PDF reports

Use the sidebar to upload one or more PDF files and start processing and indexing. Once indexing is complete, ask the 
agent questions about the reports. Uploaded documents and the index are stored in the `data` directory.

## Project structure

- `app.py` — Streamlit interface and conversation management.
- `src/agent/` — financial agent and tool-calling workflow.
- `src/tools/` — tools for market data, news, and document search.
- `src/ingestion/` — PDF processing and indexing.
- `src/storage/` — conversation and document persistence.
- `src/evaluation/` — checks whether numeric claims match available evidence.

## Limitations

Data from external services may be incomplete, delayed, or unavailable. Automated analysis and insider transaction data may contain errors.
Verify information against primary sources before making financial decisions. LucaAI is a research tool and does not provide financial advice.
