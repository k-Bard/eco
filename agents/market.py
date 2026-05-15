from skills.search import search


def run_market_agent(state: dict) -> dict:
    keyword = state["keyword"]
    results: list[dict] = []
    queries = [
        f"{keyword} 市场规模 发展趋势",
        f"{keyword} 行业分析 增长前景",
        f"{keyword} 电商 市场报告 中国",
    ]

    for query in queries:
        hits = search(query, max_results=5)
        results.extend(hits)

    overview_parts = []
    for r in results:
        overview_parts.append(f"- {r['title']}: {r['content'][:200]}...")

    overview = "\n".join(overview_parts) if overview_parts else "Insufficient data"

    return {
        "market_result": {
            "overview": overview,
            "trends": overview,
            "data_sources": [r["url"] for r in results],
            "raw_results": results,
        }
    }
