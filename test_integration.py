"""Integration test — requires valid API keys in .env. Run manually."""
import os
import sys

from dotenv import load_dotenv

load_dotenv()

if "DEEPSEEK_API_KEY" not in os.environ or "TAVILY_API_KEY" not in os.environ:
    print("SKIP: API keys not configured")
    sys.exit(0)


def test_graph_builds():
    from agents.orchestrator import build_graph

    graph = build_graph()
    assert graph is not None


def test_tavily_search():
    from skills.search import search

    results = search("test query", max_results=2)
    assert isinstance(results, list)


def test_market_agent():
    from agents.market import run_market_agent

    result = run_market_agent({"keyword": "smart watch"})
    assert "market_result" in result
    assert "overview" in result["market_result"]


def test_competitor_agent():
    from agents.competitor import run_competitor_agent

    result = run_competitor_agent({"keyword": "smart watch"})
    assert "competitor_result" in result
    assert "competitors" in result["competitor_result"]


def test_full_pipeline():
    """End-to-end: runs the full graph and checks report output."""
    from agents.orchestrator import build_graph

    graph = build_graph()
    result = graph.invoke({"keyword": "wireless earbuds"})

    assert result.get("final_report") is not None
    assert os.path.exists(result["final_report"])

    with open(result["final_report"], "r", encoding="utf-8") as f:
        content = f.read()

    assert "Product Selection" in content or "选品" in content
    assert len(content) > 500
