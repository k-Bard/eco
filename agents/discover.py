import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import (
    OpenAI,
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
    InternalServerError,
)

from skills.search import search
from agents.market import run_market_agent
from agents.competitor import run_competitor_agent

RETRYABLE_ERRORS = (
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
    InternalServerError,
)


def _call_deepseek(client: OpenAI, prompt: str) -> str:
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": prompt}],
                timeout=90.0,
            )
            return response.choices[0].message.content
        except RETRYABLE_ERRORS as e:
            if attempt == 0:
                time.sleep(3)
                continue
            raise RuntimeError(f"DeepSeek API error after retry: {e}") from e


def _extract_categories() -> list[str]:
    """Search trending e-commerce categories and extract names via DeepSeek."""
    print("[Discover] Searching trending categories...")

    queries = [
        "2026 电商 热门品类 增长最快 趋势 蓝海",
        "2026年 跨境电商 选品推荐 高利润品类",
        "2026 trending ecommerce product categories high growth China",
    ]
    all_results = []
    for q in queries:
        all_results.extend(search(q, max_results=5))

    snippets = "\n".join(
        f"- {r['title']}: {r['content'][:300]}" for r in all_results if r.get("content")
    )

    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )

    prompt = f"""You are an e-commerce market analyst. Based on the following search results about trending e-commerce product categories, identify exactly 8 specific, actionable product categories that are worth considering for cross-border or domestic e-commerce trade in 2026.

Search Results:
{snippets}

Return ONLY a numbered list of 8 categories. Each line format:
N. Category Name (Chinese) / Category Name (English) - brief reason

Example:
1. 智能穿戴 / Smart Wearables - high growth, AI integration driving demand
2. 宠物用品 / Pet Supplies - steady growth, high repurchase rate

Be specific — "智能眼镜" not "智能设备", "猫粮" not "宠物用品". Focus on categories with concrete data in the search results."""

    response = _call_deepseek(client, prompt)

    categories = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        for prefix in "0123456789":
            idx = line.find(prefix + ".")
            if idx != -1 and idx < 3:
                name = line[idx + 2:].strip()
                if " / " in name:
                    name = name.split(" / ")[0].strip()
                name = name.split("-")[0].strip().rstrip(".")
                if name and len(name) < 40:
                    categories.append(name)
                break

    print(f"[Discover] Extracted {len(categories)} categories: {categories[:8]}")
    return categories[:8]


def _scan_one(cat: str) -> dict:
    """Run market + competitor research for a single category."""
    print(f"  [Discover] Scanning: {cat}")
    try:
        m = run_market_agent({"keyword": cat})
        c = run_competitor_agent({"keyword": cat})
        return {
            "name": cat,
            "market": m.get("market_result", {}),
            "competitor": c.get("competitor_result", {}),
        }
    except Exception as e:
        print(f"  [Discover] Scan failed for '{cat}': {e}")
        return {
            "name": cat,
            "market": {"overview": "Scan failed", "data_sources": []},
            "competitor": {"competitors": "Scan failed", "price_range": "", "key_features": ""},
        }


def run_discover_agent(state: dict) -> dict:
    categories = _extract_categories()

    print(f"\n[Discover] Parallel scanning {len(categories)} categories...")
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_scan_one, cat): cat for cat in categories}
        for f in as_completed(futures):
            results.append(f.result())

    ordered = sorted(results, key=lambda r: categories.index(r["name"]) if r["name"] in categories else 99)
    return {"categories": ordered}
