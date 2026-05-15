from skills.search import search


def run_competitor_agent(state: dict) -> dict:
    keyword = state.get("keyword")
    if not keyword or not keyword.strip():
        raise ValueError("state must contain a non-empty 'keyword'")

    results: list[dict] = []
    queries = [
        f"{keyword} 头部品牌 价格区间",
        f"{keyword} 主流产品 产品对比",
        f"{keyword} 热销 卖点分析",
    ]

    for query in queries:
        hits = search(query, max_results=5)
        results.extend(hits)

    parts = []
    for r in results:
        title = r.get("title", "")
        content = r.get("content", "")
        if title and content:
            parts.append(f"- {title}: {content[:200]}...")

    competitors = "\n".join(parts) if parts else "Insufficient data"

    return {
        "competitor_result": {
            "competitors": competitors,
            "price_range": competitors,
            "key_features": competitors,
            "raw_results": results,
        }
    }
