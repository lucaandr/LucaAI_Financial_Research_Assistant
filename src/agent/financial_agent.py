import os
import time
import logging

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.tools.market_tools import (
    get_analyst_ratings,
    get_insider_activity,
    get_stock_financials,
    get_technical_analysis,
    search_financial_docs,
    search_latest_web_news,
)
from src.evaluation.faithfulness import evaluate_faithfulness

load_dotenv()

#  Logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/agent.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("financial_agent")

# --- Cost tracking (Groq pricing per 1M tokens, model qwen/qwen3.6-27b) ----
PRICE_PER_M_INPUT = 0.60
PRICE_PER_M_OUTPUT = 3.00
MAX_TOOL_ITERATIONS = 5
# lowered from 6: even with per-message truncation below, a long
MAX_HISTORY_MESSAGES = 4
# testing session can still accumulate more than the per-request
# budget allows — trading a bit of context depth for headroom
# circuit breaker: after this many failures, the tool is disabled for the rest of the session
MAX_TOOL_FAILURES = 2
# cap on how much of a single tool's raw output gets fed back
MAX_TOOL_OUTPUT_CHARS = 1500
# to the model — without this, 3 tool calls in one turn
# (financials + technical + web news, each already a few
# hundred tokens) can push a single request over the
# provider's per-request token limit
# cap on how much of EACH stored message from earlier turns
MAX_HISTORY_MESSAGE_CHARS = 500
# gets re-sent as context. Limiting only the *count* of
# history messages (MAX_HISTORY_MESSAGES) wasn't enough —
# a single old message can itself be a 1000+ token
# structured analysis, so a handful of them alone can
# exceed the per-request budget. The full text is still
# shown to the user in the UI; only what's re-sent to the
# model as prior context gets shortened.


def estimate_cost(usage) -> float:
    """Estimates the cost (USD) based on the response's usage_metadata."""
    if not usage:
        return 0.0
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    return (input_tokens / 1_000_000 * PRICE_PER_M_INPUT) + \
           (output_tokens / 1_000_000 * PRICE_PER_M_OUTPUT)


@tool
def get_stock_financials_tool(ticker_symbol: str) -> str:
    """Fetches real-time stock market data (current price, Market Cap, P/E ratio),
    trading volume (today's volume, average volume, relative volume vs average),
    profitability margins, YoY growth, and financial health metrics (debt-to-equity,
    ROE, free cash flow). The argument must be the stock ticker symbol, e.g.: MU, NVDA, AAPL."""
    return get_stock_financials(ticker_symbol)


@tool
def get_insider_activity_tool(ticker_symbol: str) -> str:
    """Fetches recent insider (executives, directors) buy/sell transactions for a
    stock — combining news coverage (generally reliable for correct buy/sell
    classification) with automated SEC Form 4 parsing (broader coverage, but
    known to occasionally misclassify complex filings). If the two disagree,
    the result explicitly flags it. Use this when the user asks about insider
    buying/selling, insider confidence, or recent insider activity/news.
    The argument must be the stock ticker symbol, e.g.: MU, NVDA, AAPL."""
    return get_insider_activity(ticker_symbol)


@tool
def get_analyst_ratings_tool(ticker_symbol: str) -> str:
    """Fetches Wall Street analyst consensus (buy/hold/sell rating) and price
    targets (low/mean/median/high) for a stock. Use this together with
    get_stock_financials_tool and get_technical_analysis_tool when the user
    asks whether a stock is a good buy, or wants a recommendation.
    The argument must be the stock ticker symbol, e.g.: MU, NVDA, AAPL."""
    return get_analyst_ratings(ticker_symbol)


@tool
def get_technical_analysis_tool(ticker_symbol: str) -> str:
    """Computes technical indicators for a stock based on 1 year of price history:
    moving averages (SMA20/50/200), golden/death cross trend signal, RSI(14)
    overbought/oversold signal, 30-day annualized volatility, and position within
    the 52-week range. Use this when the user asks for technical analysis, trend,
    momentum, or whether a stock looks overbought/oversold.
    The argument must be the stock ticker symbol, e.g.: MU, NVDA, AAPL."""
    return get_technical_analysis(ticker_symbol)


@tool
def search_financial_docs_tool(query: str) -> str:
    """Searches historical data, revenue, margins, and results from 10-K/10-Q financial
    reports the user has uploaded and indexed (stored in ChromaDB). ALWAYS call this
    tool — never answer from general knowledge — whenever the user references "the
    report", "the filing", "the document/PDF", "according to the 10-Q/10-K", or
    similar, even if you believe you already know the answer. This is the only tool
    that can see the actual uploaded document; general knowledge about a company is
    not a substitute and may not match what's in the specific filing the user has in
    mind. If this tool returns no relevant results, say so explicitly rather than
    silently falling back to general knowledge.
    The argument is a question or search term."""
    return search_financial_docs(query)


@tool
def search_latest_web_news_tool(query: str) -> str:
    """Searches the internet for the latest financial news, recent quarterly reports
    (Q1, Q2, Q3), or current events not found in the local PDFs.
    The argument is a search term (e.g. 'Micron latest quarterly earnings report')."""
    return search_latest_web_news(query)


TOOLS = [
    get_stock_financials_tool,
    get_technical_analysis_tool,
    get_analyst_ratings_tool,
    get_insider_activity_tool,
    search_financial_docs_tool,
    search_latest_web_news_tool,
]
tools_map = {t.name: t for t in TOOLS}


class _LLMCallFailed(Exception):
    """Internal signal used to unwind out of the tool-calling loop with a
    user-friendly message when a Groq API call fails, instead of letting the
    raw exception crash the app."""
    pass


class FinancialAgent:
    """
    Financial agent with tool-calling, conversation memory, a per-tool
    circuit breaker, and session-level cost tracking.

    One instance = one session (keeps state across queries: accumulated
    cost, per-tool failures). In a multi-user app (e.g. Streamlit),
    keep one instance per user (see app.py, st.session_state).
    """

    def __init__(self, max_tool_iterations: int = MAX_TOOL_ITERATIONS,
                 max_tool_failures: int = MAX_TOOL_FAILURES):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Check your local .env file, "
                "or (in Docker) make sure the container was started with "
                "--env-file .env / -e GROQ_API_KEY=..."
            )

        self.llm = ChatGroq(
            model="qwen/qwen3.6-27b",
            groq_api_key=api_key,
            temperature=0,
            max_tokens=1000,
            reasoning_effort="none",
            model_kwargs={
                "parallel_tool_calls": True,
            },
            reasoning_format="parsed",
        )
        self.llm_with_tools = self.llm.bind_tools(TOOLS)

        self.max_tool_iterations = max_tool_iterations
        self.max_tool_failures = max_tool_failures

        # Circuit breaker: per-tool failures, persists for the session
        self.tool_failure_counts: dict[str, int] = {}
        self.disabled_tools: set[str] = set()

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.query_count = 0
        # Exposed to the UI after each run. Tool output is retained only for
        # this turn so the score cannot accidentally use stale evidence.
        self.last_tool_outputs: list[dict[str, str]] = []
        self.last_evaluation: dict = {}
        self.last_iterations = 0

    #  circuit breaker

    def _call_tool_safely(self, tool_name: str, tool_args: dict) -> str:
        if tool_name in self.disabled_tools:
            logger.warning(
                "Tool %s is disabled (circuit breaker) — skipping call", tool_name)
            return f"⚠️ Tool '{tool_name}' has failed {self.max_tool_failures} times this session and is temporarily disabled."

        if tool_name not in tools_map:
            return f"⚠️ Error: tool '{tool_name}' does not exist."

        t0 = time.time()
        logger.info("  -> Calling %s(%s)", tool_name, tool_args)
        try:
            output = str(tools_map[tool_name].invoke(tool_args))
            if len(output) > MAX_TOOL_OUTPUT_CHARS:
                logger.warning(
                    "  Tool %s output truncated from %d to %d chars to stay under the token budget",
                    tool_name, len(output), MAX_TOOL_OUTPUT_CHARS,
                )
                output = output[:MAX_TOOL_OUTPUT_CHARS] + \
                    "\n[...output truncated to fit the token budget]"
            logger.info("  <- %s responded in %.2fs",
                        tool_name, time.time() - t0)
            # Reset consecutive failure count on a successful call
            self.tool_failure_counts[tool_name] = 0
            return output
        except Exception:
            logger.exception("Error while executing tool %s", tool_name)
            failures = self.tool_failure_counts.get(tool_name, 0) + 1
            self.tool_failure_counts[tool_name] = failures
            if failures >= self.max_tool_failures:
                self.disabled_tools.add(tool_name)
                logger.warning(
                    "Tool %s failed %d times — disabled for the rest of the session",
                    tool_name, failures,
                )
            return f"⚠️ Internal error while executing tool {tool_name}."

    def _record_usage(self, response) -> None:
        usage = getattr(response, "usage_metadata", None) or {}
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += estimate_cost(usage)

    def session_stats(self) -> str:
        """Cost/usage summary for the current session (the 'cost' command)."""
        total_tokens = self.total_input_tokens + self.total_output_tokens
        disabled = ", ".join(sorted(self.disabled_tools)) or "none"
        return (
            "📊 Session statistics:\n"
            f"- Queries processed: {self.query_count}\n"
            f"- Input tokens: {self.total_input_tokens}\n"
            f"- Output tokens: {self.total_output_tokens}\n"
            f"- Total tokens: {total_tokens}\n"
            f"- Estimated cost (Groq, qwen/qwen3.6-27b): ${self.total_cost:.6f} USD\n"
            f"- Disabled tools (circuit breaker): {disabled}"
        )

    # safe LLM invocation

    def _invoke_llm_safely(self, messages):
        """Wraps every call to Groq. Without this, any API-level error (rate
        limits, request-too-large, network issues) propagates all the way up
        and crashes the whole Streamlit app with a raw traceback instead of
        showing the user a clear message."""
        try:
            return self.llm_with_tools.invoke(messages)
        except Exception as e:
            error_text = str(e)

            if "output tokens per minute" in error_text.lower() or "otpm" in error_text.lower():
                logger.warning(
                    "OTPM limit hit, retrying once with a reduced max_tokens")
                try:
                    return self.llm_with_tools.bind(max_tokens=350).invoke(messages)
                except Exception:
                    logger.exception("Reduced-token retry also failed")
            logger.exception("LLM call failed")
            error_text = str(e)
            if "413" in error_text or "too large" in error_text.lower():
                friendly = (
                    "⚠️ This request became too large for the model's per-request token limit "
                    "(this can happen after a long conversation with several detailed analyses). "
                    "Try asking a shorter, more specific question, or start a new conversation."
                )
            elif "429" in error_text or "rate_limit" in error_text.lower():
                friendly = "⚠️ The API rate limit was hit and retries were exhausted. Please wait a moment and try again."
            else:
                friendly = "⚠️ The AI service returned an error while processing this request. Please try again."
            raise _LLMCallFailed(friendly) from e

    def run(self, query: str, history: list | None = None) -> str:
        """
        history: list of previous conversation messages, formatted as
        [{"role": "user"/"assistant", "content": "..."}], in chronological
        order (excluding the current message — that's passed separately via `query`).
        """
        if query.strip().lower() == "cost":
            return self.session_stats()

        start_time = time.time()
        self.query_count += 1
        self.last_tool_outputs = []
        self.last_evaluation = {}
        self.last_iterations = 0
        cost_before = self.total_cost
        logger.info("New query: %r (history: %d messages)",
                    query, len(history or []))

        messages = [
            SystemMessage(content=(
                "You are an expert financial analysis assistant. Use the available tools "
                "to gather information, then synthesize a clear, detailed, and well-structured "
                "answer in English. Use get_stock_financials_tool for current price, valuation, "
                "profitability, growth, and financial health. Use get_technical_analysis_tool "
                "when the user asks about trend, momentum, chart patterns, or whether a stock "
                "looks overbought/oversold. Use get_analyst_ratings_tool for Wall Street consensus "
                "and price targets. Use get_insider_activity_tool for insider buying/selling "
                "questions — it already cross-checks news coverage against automated SEC filing "
                "data internally and will flag any disagreement between the two. Trading volume "
                "is already included in get_stock_financials_tool, so don't call a separate tool "
                "for volume.\n\n"
                "If the user asks whether a stock is a buy, hold, or sell, or asks for a "
                "recommendation: request get_stock_financials_tool, get_technical_analysis_tool, "
                "and get_analyst_ratings_tool ALL AT ONCE, in the same turn, instead of one at a "
                "time — this avoids unnecessary round trips and rate-limit delays. Then write a "
                "'Recommendation' section that synthesizes all three. Base a suggested entry zone on the technical data "
                "(e.g. near a moving average acting as support, or the lower part of the 52-week "
                "range) and reference the analyst price target range as context for potential "
                "upside/downside. Always end this section with a short disclaimer that this is "
                "an analysis based on available data, not personalized financial advice, and that "
                "the user should do their own research or consult a licensed financial advisor "
                "before making investment decisions.\n\n"
                "CRITICAL: whenever the user's question references 'the report', 'the filing', "
                "'the 10-Q/10-K', 'the document/PDF', or similar — you MUST call "
                "search_financial_docs_tool before answering, even if you think you already know "
                "the answer from general knowledge about the company. General knowledge is not a "
                "substitute for the specific document the user has uploaded, and numbers can "
                "differ. Never answer a document-referencing question from memory alone.\n\n"
                "You have access to the conversation history — if the user refers to something "
                "discussed earlier (e.g. 'this report', 'that company'), use the context from "
                "previous messages to understand what they mean."
                "If a document search does not return a specific line item like Total Revenue, try searching specifically for 'Condensed Consolidated Statements of Income' or 'Total revenue' once.  "
                "If still not found, summarize the available metrics without retrying endlessly."
            )),
        ]

        for msg in (history or [])[-MAX_HISTORY_MESSAGES:]:
            content = msg["content"]
            if len(content) > MAX_HISTORY_MESSAGE_CHARS:
                content = content[:MAX_HISTORY_MESSAGE_CHARS] + \
                    " [...earlier answer truncated for context]"
            if msg["role"] == "user":
                messages.append(HumanMessage(content=content))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=content))

        messages.append(HumanMessage(content=query))

        try:
            response = self._invoke_llm_safely(messages)
        except _LLMCallFailed as e:
            return str(e)
        self._record_usage(response)

        iterations = 0
        while response.tool_calls:
            iterations += 1
            self.last_iterations = iterations
            if iterations > self.max_tool_iterations:
                logger.warning(
                    "Reached the %d tool-calling iteration limit for query %r — stopping the loop.",
                    self.max_tool_iterations, query,
                )
                messages.append(ToolMessage(
                    content="The maximum number of consecutive tool calls was reached. Answer with what you already have.",
                    tool_call_id=response.tool_calls[0]["id"],
                ))
                try:
                    response = self._invoke_llm_safely(messages)
                except _LLMCallFailed as e:
                    return str(e)
                self._record_usage(response)
                break

            logger.info("Agent is calling %d tool(s) (iteration %d)",
                        len(response.tool_calls), iterations)
            messages.append(response)

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]

                tool_output = self._call_tool_safely(tool_name, tool_args)
                self.last_tool_outputs.append({
                    "tool": tool_name,
                    "output": str(tool_output),
                })

                messages.append(ToolMessage(
                    content=str(tool_output),
                    tool_call_id=tool_id,
                ))

            try:
                response = self._invoke_llm_safely(messages)
            except _LLMCallFailed as e:
                return str(e)
            self._record_usage(response)

        elapsed = time.time() - start_time
        query_cost = self.total_cost - cost_before
        logger.info(
            "Query finished in %.2fs, query cost $%.6f (session total $%.6f), %d tool-calling iterations",
            elapsed, query_cost, self.total_cost, iterations,
        )

        final_content = (response.content or "").strip()
        if not final_content:
            # Fallback: the model put the entire answer in reasoning_content instead of content
            # (happens with reasoning models on short follow-up queries).
            reasoning = (response.additional_kwargs or {}
                         ).get("reasoning_content", "")
            if reasoning:
                logger.warning(
                    "content was empty, using reasoning_content as fallback for %r", query)
                final_content = reasoning.strip()
            else:
                logger.warning(
                    "content was empty and no reasoning_content for %r", query)
                final_content = "⚠️ I couldn't generate an answer for this question. Could you rephrase it or try again?"

        self.last_evaluation = evaluate_faithfulness(
            final_content, self.last_tool_outputs).as_dict()
        return final_content


_default_agent = FinancialAgent()


def run_financial_assistant(query: str, history: list | None = None) -> str:
    return _default_agent.run(query, history=history)


if __name__ == "__main__":
    print("--- Financial Agent Test ---")
    agent = FinancialAgent()
    user_query = (
        "What was Micron's revenue and net profit in 2025 according to the report, "
        "and what is the current stock price for ticker MU?"
    )
    result = agent.run(user_query)
    print("\nFINAL ANSWER:\n")
    print(result)
    print("\n" + agent.session_stats())
