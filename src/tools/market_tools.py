import os
import logging

from firecrawl import FirecrawlApp
from dotenv import load_dotenv
import pandas as pd
import yfinance as yf
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

logger = logging.getLogger("market_tools")

BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
DB_DIR = os.path.join(BASE_DIR, "data", "chromadb")


def _fmt_pct(value) -> str:
    return f"{value * 100:.2f}%" if isinstance(value, (int, float)) else "N/A"


def _fmt_money(value) -> str:
    return f"${value:,.0f}" if isinstance(value, (int, float)) else "N/A"


def _fmt_ratio(value) -> str:
    return f"{value:.2f}" if isinstance(value, (int, float)) else "N/A"


def get_stock_financials(ticker_symbol: str) -> str:
    """Fetch live financial data for a company using its stock ticker (e.g. MU for Micron, NVDA, AAPL)."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info

        current_price = info.get("currentPrice") or info.get(
            "regularMarketPrice", "N/A")
        market_cap = info.get("marketCap", "N/A")
        pe_ratio = info.get("trailingPE", "N/A")
        fifty_two_high = info.get("fiftyTwoWeekHigh", "N/A")
        fifty_two_low = info.get("fiftyTwoWeekLow", "N/A")
        volume = info.get("volume") or info.get("regularMarketVolume")
        avg_volume_10d = info.get("averageDailyVolume10Day")
        avg_volume_3m = info.get("averageVolume")

        market_cap_str = (
            f"${market_cap:,}" if isinstance(market_cap, (int, float))
            else str(market_cap)
        )

        volume_str = f"{volume:,}" if isinstance(
            volume, (int, float)) else "N/A"
        avg_volume_10d_str = f"{avg_volume_10d:,}" if isinstance(
            avg_volume_10d, (int, float)) else "N/A"
        avg_volume_3m_str = f"{avg_volume_3m:,}" if isinstance(
            avg_volume_3m, (int, float)) else "N/A"

        relative_volume = None
        if isinstance(volume, (int, float)) and isinstance(avg_volume_3m, (int, float)) and avg_volume_3m:
            relative_volume = volume / avg_volume_3m

        result = (
            f"📊 Live data for {ticker_symbol.upper()}:\n"
            f"- Sector: {info.get('sector', 'N/A')} | Industry: {info.get('industry', 'N/A')}\n"
            f"- Current price: ${current_price}\n"
            f"- Market Cap: {market_cap_str}\n"
            f"- Trailing P/E: {pe_ratio}\n"
            f"- 52-week high: ${fifty_two_high}\n"
            f"- 52-week low: ${fifty_two_low}\n"
            f"\n📦 Trading volume:\n"
            f"- Today's volume: {volume_str} shares\n"
            f"- Average volume (10 days): {avg_volume_10d_str} shares\n"
            f"- Average volume (3 months): {avg_volume_3m_str} shares\n"
            + (f"- Relative volume: {relative_volume:.2f}x average"
               + (" (unusually high activity)" if relative_volume >= 1.5 else
                  " (unusually low activity)" if relative_volume <= 0.5 else "") + "\n"
               if relative_volume is not None else "")
            + f"\n💰 Profitability:\n"
            f"- Gross margin: {_fmt_pct(info.get('grossMargins'))}\n"
            f"- Operating margin: {_fmt_pct(info.get('operatingMargins'))}\n"
            f"- Net profit margin: {_fmt_pct(info.get('profitMargins'))}\n"
            f"- Return on equity (ROE): {_fmt_pct(info.get('returnOnEquity'))}\n"
            f"\n📈 Growth (YoY):\n"
            f"- Revenue growth: {_fmt_pct(info.get('revenueGrowth'))}\n"
            f"- Earnings growth: {_fmt_pct(info.get('earningsGrowth'))}\n"
            f"\n🏦 Financial health:\n"
            f"- Debt-to-equity: {_fmt_ratio(info.get('debtToEquity'))}\n"
            f"- Current ratio: {_fmt_ratio(info.get('currentRatio'))}\n"
            f"- Free cash flow: {_fmt_money(info.get('freeCashflow'))}\n"
        )
        logger.info("get_stock_financials OK for %s", ticker_symbol)
        return result
    except Exception:
        logger.exception(
            "Error fetching stock data for %s", ticker_symbol)
        return f"⚠️ Error fetching stock data for {ticker_symbol}."


def get_analyst_ratings(ticker_symbol: str) -> str:
    """Fetches Wall Street analyst consensus and price targets for a stock
    (rating, number of analysts, low/mean/median/high price targets)."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info

        recommendation_key = info.get("recommendationKey", "N/A")
        recommendation_mean = info.get("recommendationMean")
        num_analysts = info.get("numberOfAnalystOpinions", "N/A")
        target_low = info.get("targetLowPrice")
        target_mean = info.get("targetMeanPrice")
        target_median = info.get("targetMedianPrice")
        target_high = info.get("targetHighPrice")
        current_price = info.get(
            "currentPrice") or info.get("regularMarketPrice")

        result = (
            f"👨‍💼 Analyst ratings for {ticker_symbol.upper()}:\n"
            f"- Consensus rating: {str(recommendation_key).upper()} "
            f"(score: {_fmt_ratio(recommendation_mean)}, scale 1=Strong Buy to 5=Strong Sell)\n"
            f"- Number of analysts covering: {num_analysts}\n"
            f"- Price targets — Low: {_fmt_money(target_low)} | Mean: {_fmt_money(target_mean)} | "
            f"Median: {_fmt_money(target_median)} | High: {_fmt_money(target_high)}\n"
        )
        if isinstance(target_mean, (int, float)) and isinstance(current_price, (int, float)) and current_price:
            upside = (target_mean - current_price) / current_price * 100
            result += f"- Implied upside/downside vs current price (${current_price}): {upside:+.1f}%\n"

        logger.info("get_analyst_ratings OK for %s", ticker_symbol)
        return result
    except Exception:
        logger.exception(
            "Error fetching analyst ratings for %s", ticker_symbol)
        return f"⚠️ Error fetching analyst ratings for {ticker_symbol}."


def _row_get(row, *candidate_names):
    """yfinance's insider_transactions column names have shifted between
    versions in the past. Try each candidate name in order instead of
    hardcoding one, so this doesn't silently return blank fields if the
    installed yfinance version uses slightly different column labels."""
    for name in candidate_names:
        if name in row.index:
            return row[name]
    return None


def get_insider_activity(ticker_symbol: str, max_transactions: int = 8) -> str:
    """Fetches recent insider (executives, directors) buy/sell transactions for a
    stock, as reported to the SEC. Useful for spotting insider buying/selling
    trends that might signal confidence or concern about the company's outlook."""
    sections = []

    try:
        news_result = search_latest_web_news(
            f"{ticker_symbol} insider buying selling shares recent SEC Form 4"
        )
        sections.append(
            "📰 From recent news coverage (generally more reliable for "
            "correctly classifying buy vs sell on complex filings):\n" + news_result
        )
    except Exception:
        logger.exception(
            "Insider news cross-check failed for %s", ticker_symbol)

    try:
        ticker = yf.Ticker(ticker_symbol)
        transactions = ticker.insider_transactions

        if transactions is None or transactions.empty:
            logger.info(
                "get_insider_activity: no yfinance data for %s", ticker_symbol)
        else:
            logger.info("get_insider_activity: columns for %s: %s",
                        ticker_symbol, list(transactions.columns))

            date_col = next(
                (c for c in ["Start Date", "Date"] if c in transactions.columns), None)
            if date_col:
                transactions = transactions.copy()
                transactions[date_col] = pd.to_datetime(
                    transactions[date_col], errors="coerce")
                transactions = transactions.sort_values(
                    date_col, ascending=False)
            else:
                logger.warning(
                    "get_insider_activity: no recognizable date column for %s, cannot guarantee recency ordering", ticker_symbol)

            recent = transactions.head(max_transactions)

            lines = [
                f"👤 Recent insider activity for {ticker_symbol.upper()} (automated SEC Form 4 parsing):"]
            buy_count, sell_count = 0, 0
            total_buy_value, total_sell_value = 0, 0

            for _, row in recent.iterrows():
                insider = _row_get(row, "Insider", "Filer", "Name") or "N/A"
                position = _row_get(
                    row, "Position", "Relation", "Title") or "N/A"
                transaction_type = _row_get(
                    row, "Transaction", "Text", "Type") or "N/A"
                shares = _row_get(row, "Shares")
                value = _row_get(row, "Value")
                start_date = _row_get(row, "Start Date", "Date") or "N/A"
                if hasattr(start_date, "strftime"):
                    start_date = start_date.strftime("%Y-%m-%d")

                shares_str = f"{shares:,.0f}" if isinstance(
                    shares, (int, float)) else "N/A"
                value_str = _fmt_money(value) if isinstance(
                    value, (int, float)) else "N/A"

                lines.append(
                    f"- {start_date} | {insider} ({position}): {transaction_type} — "
                    f"{shares_str} shares, {value_str}"
                )

                transaction_lower = str(transaction_type).lower()
                if "buy" in transaction_lower or "purchase" in transaction_lower:
                    buy_count += 1
                    if isinstance(value, (int, float)):
                        total_buy_value += value
                elif "sale" in transaction_lower or "sell" in transaction_lower:
                    sell_count += 1
                    if isinstance(value, (int, float)):
                        total_sell_value += value

            buy_summary = ("no purchases found in this data source" if buy_count == 0
                           else f"{buy_count} buy(s) totaling {_fmt_money(total_buy_value)}")
            lines.append(
                f"\nSummary of these {len(recent)} most recent transactions from this source: "
                f"{buy_summary}, {sell_count} sale(s) totaling {_fmt_money(total_sell_value)}."
            )
            lines.append(
                "\n⚠️ KNOWN RELIABILITY ISSUE: this automated parsing has been confirmed wrong on "
                "at least one real case — a filing combining a genuine purchase with a share "
                "transfer to a family trust/partnership had the purchase mislabeled as a sale. "
                "If this disagrees with the news section above, trust the news section."
            )

            sections.append(
                "\n\n📊 From automated SEC Form 4 parsing (secondary, less reliable — see warning below):\n"
                + "\n".join(lines)
            )
    except Exception:
        logger.exception(
            "Error fetching yfinance insider activity for %s", ticker_symbol)
        sections.append(
            "\n\n📊 Automated SEC Form 4 parsing was unavailable for this query.")

    if not sections:
        return f"No insider activity data available for {ticker_symbol.upper()} from any source."

    logger.info("get_insider_activity OK for %s", ticker_symbol)
    return "\n".join(sections)


def get_technical_analysis(ticker_symbol: str) -> str:
    """Computes basic technical indicators (moving averages, RSI, volatility,
    52-week range position) for a stock, based on 1 year of daily price history."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="1y")

        if hist.empty or len(hist) < 60:
            return f"⚠️ Not enough historical price data for {ticker_symbol} to compute technical indicators."

        close = hist["Close"]

        sma20 = close.rolling(window=20).mean()
        sma50 = close.rolling(window=50).mean()
        sma200 = close.rolling(window=200).mean() if len(
            close) >= 200 else None

        last_close = close.iloc[-1]
        last_sma20 = sma20.iloc[-1]
        last_sma50 = sma50.iloc[-1]

        trend_signal = "N/A (not enough history for a 200-day average)"
        if sma200 is not None and not sma50.iloc[-6:].isna().any() and not sma200.iloc[-6:].isna().any():
            prev_diff = sma50.iloc[-6] - sma200.iloc[-6]
            curr_diff = sma50.iloc[-1] - sma200.iloc[-1]
            if prev_diff < 0 and curr_diff > 0:
                trend_signal = "🟢 Golden Cross detected recently (bullish signal: SMA50 crossed above SMA200)"
            elif prev_diff > 0 and curr_diff < 0:
                trend_signal = "🔴 Death Cross detected recently (bearish signal: SMA50 crossed below SMA200)"
            elif curr_diff > 0:
                trend_signal = "SMA50 above SMA200 — established uptrend"
            else:
                trend_signal = "SMA50 below SMA200 — established downtrend"

        # RSI(14)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        last_rsi = rsi.iloc[-1]
        if last_rsi >= 70:
            rsi_signal = "Overbought"
        elif last_rsi <= 30:
            rsi_signal = "Oversold"
        else:
            rsi_signal = "Neutral"

        returns = close.pct_change().dropna()
        daily_vol = returns.iloc[-30:].std()
        annualized_vol = daily_vol * (252 ** 0.5) * 100

        fifty_two_high = close.max()
        fifty_two_low = close.min()
        range_position = (
            (last_close - fifty_two_low) /
            (fifty_two_high - fifty_two_low) * 100
            if fifty_two_high != fifty_two_low else None
        )

        result = (
            f"📉 Technical analysis for {ticker_symbol.upper()} (1-year lookback):\n"
            f"- Current price: ${last_close:.2f}\n"
            f"- SMA20: ${last_sma20:.2f} | SMA50: ${last_sma50:.2f}"
            + (f" | SMA200: ${sma200.iloc[-1]:.2f}\n" if sma200 is not None else "\n")
            + f"- Trend signal: {trend_signal}\n"
            f"- RSI(14): {last_rsi:.1f} ({rsi_signal})\n"
            f"- 30-day annualized volatility: {annualized_vol:.1f}%\n"
        )
        if range_position is not None:
            result += f"- Position in 52-week range: {range_position:.1f}% (0% = 52w low, 100% = 52w high)\n"

        logger.info("get_technical_analysis OK for %s", ticker_symbol)
        return result
    except Exception:
        logger.exception(
            "Error computing technical analysis for %s", ticker_symbol)
        return f"⚠️ Error computing technical analysis for {ticker_symbol}."


_embeddings = None
_vector_db = None


def _get_vector_db():
    """Lazily initializes the embeddings model and Chroma store once per process,
    instead of on every call. Re-creating HuggingFaceEmbeddings each time triggers
    a full round of Hugging Face Hub cache-check requests (~4-5s of pure network
    overhead per query), even though the model is already downloaded locally."""
    global _embeddings, _vector_db
    if _vector_db is None:
        logger.info(
            "Initializing embeddings model and Chroma vector store (first RAG call this session)...")
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        _vector_db = Chroma(persist_directory=DB_DIR,
                            embedding_function=_embeddings)
    return _vector_db


def search_financial_docs(query: str, k: int = 3) -> str:
    """Search for relevant information in the locally indexed financial PDF reports (ChromaDB)."""
    try:
        vector_db = _get_vector_db()
        results = vector_db.similarity_search(query, k=k)

        if not results:
            logger.info("search_financial_docs: no results for %r", query)
            return "No relevant information found in the stored reports."

        context = []
        for i, doc in enumerate(results, 1):
            source = os.path.basename(doc.metadata.get("source", "PDF"))
            page = doc.metadata.get("page", "N/A")
            context.append(
                f"--- Result {i} (Source: {source}, Page: {page}) ---\n{doc.page_content}")

        logger.info("search_financial_docs: %d results for %r",
                    len(results), query)
        return "\n\n".join(context)
    except Exception:
        logger.exception("Error searching the vector database for %r", query)
        return "⚠️ Error searching the vector database."


def _extract_firecrawl_items(response) -> list:
    """Normalizes the Firecrawl v2 response. The Python SDK returns the data
    object directly (e.g. {"web": [...], "news": [...]}), NOT wrapped in an
    extra "data" field — that extra nesting only exists in the raw REST
    response. This checks both shapes so it keeps working across SDK versions."""
    if response is None:
        return []

    if isinstance(response, dict):
        if response.get("web") or response.get("news"):
            return response.get("web") or response.get("news") or []
        nested = response.get("data")
        if isinstance(nested, dict):
            return nested.get("web") or nested.get("news") or []
        return []

    web = getattr(response, "web", None)
    news = getattr(response, "news", None)
    if web or news:
        return web or news or []

    nested = getattr(response, "data", None)
    if nested is None:
        return []
    if isinstance(nested, dict):
        return nested.get("web") or nested.get("news") or []
    return getattr(nested, "web", None) or getattr(nested, "news", None) or []


def search_latest_web_news(query: str) -> str:
    """Search the web for the latest financial news and recent reports using Firecrawl,
    with a DuckDuckGo (ddgs) fallback if Firecrawl is unavailable or fails."""
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if api_key:
        try:
            app = FirecrawlApp(api_key=api_key)
            search_result = app.search(query, limit=3)
            items = _extract_firecrawl_items(search_result)

            results = []
            for item in items:
                if isinstance(item, dict):
                    title = item.get("title", "No title")
                    description = item.get("description", "")
                    url = item.get("url", "")
                else:
                    title = getattr(item, "title", "No title")
                    description = getattr(item, "description", "")
                    url = getattr(item, "url", "")
                results.append(
                    f"Title: {title}\nSource: {url}\nSummary: {description}\n")

            if results:
                logger.info("Firecrawl: %d results for %r",
                            len(results), query)
                return "\n---\n".join(results)
            logger.info(
                "Firecrawl: no results for %r, falling back to DuckDuckGo (raw response type: %s, preview: %.200s)",
                query, type(search_result).__name__, str(search_result),
            )
        except Exception:
            logger.exception(
                "Firecrawl failed for %r, falling back to DuckDuckGo", query)
    else:
        logger.warning(
            "FIRECRAWL_API_KEY missing from .env — going straight to DuckDuckGo")

    try:
        from ddgs import DDGS
        from ddgs.exceptions import RatelimitException

        with DDGS() as ddgs:
            results_ddg = list(ddgs.text(query, max_results=3))

        if not results_ddg:
            logger.info("DuckDuckGo: no results for %r", query)
            return "No recent results found on the web."

        results = [
            f"Title: {r.get('title', 'No title')}\nSource: {r.get('href', '')}\nSummary: {r.get('body', '')}"
            for r in results_ddg
        ]
        logger.info("DuckDuckGo: %d results for %r", len(results), query)
        return "\n---\n".join(results)
    except RatelimitException:
        logger.warning("DuckDuckGo rate limit hit for %r", query)
        return "⚠️ Web search is temporarily rate-limited. Answer using the data already available."
    except Exception:
        logger.exception("DuckDuckGo failed for %r", query)
        return "⚠️ Could not fetch live news from the web. Answer using the data already available."


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("--- Test 1: Live Stock Data (yfinance) ---")
    print(get_stock_financials("MU"))

    print("\n--- Test 2: Semantic RAG Search (ChromaDB) ---")
    print(search_financial_docs("revenue and net income"))
