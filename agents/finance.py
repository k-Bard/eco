import os
import re
import time
from datetime import datetime

from openai import (
    OpenAI,
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
    InternalServerError,
)

from skills.search import search

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


def _extract_companies(client: OpenAI, keyword: str, market: dict, competitor: dict) -> list[dict]:
    print("  [Finance] Extracting listed companies from research...")

    raw = (
        f"Market overview: {market.get('overview', '')[:1500]}\n\n"
        f"Competitor data: {competitor.get('competitors', '')[:1500]}"
    )

    prompt = f"""From the following e-commerce product category research for "{keyword}", extract up to 8 publicly traded companies (A-share, HK, US-listed Chinese companies) that are key players in this industry's supply chain.

Research data:
{raw}

Return ONLY a JSON array. Each item: {{"name": "公司中文名", "code": "股票代码", "exchange": "交易所", "role": "产业链角色"}}.

Example:
[
  {{"name": "歌尔股份", "code": "002241", "exchange": "深交所", "role": "声学组件及整机代工"}},
  {{"name": "立讯精密", "code": "002475", "exchange": "深交所", "role": "连接器及模组供应商"}}
]

If you cannot identify any specific public company from the data, return an empty array [].
Return ONLY the JSON, no other text."""

    response = _call_deepseek(client, prompt)

    json_match = re.search(r"\[.*\]", response, re.DOTALL)
    if not json_match:
        return []

    import json
    try:
        companies = json.loads(json_match.group())
    except json.JSONDecodeError:
        return []

    print(f"  [Finance] Found {len(companies)} companies: {[c['name'] for c in companies]}")
    return companies[:8]


def _search_company_financials(company: dict) -> dict:
    name = company["name"]
    code = company.get("code", "")
    queries = [
        f"{name} {code} 股票 2026 营收 净利润 增长 估值",
        f"{name} {code} 研报 投资分析 目标价",
    ]
    all_results = []
    for q in queries:
        all_results.extend(search(q, max_results=3))

    snippets = " | ".join(
        f"{r['title']}: {r['content'][:250]}" for r in all_results if r.get("content")
    )
    result = dict(company)
    result["financial_snippets"] = snippets[:2000] if snippets else "Insufficient data"
    return result


def run_finance_agent(state: dict) -> dict:
    print("\n[Finance] Starting investment analysis...")

    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )

    keyword = state["keyword"]
    market = state.get("market_result") or {}
    competitor = state.get("competitor_result") or {}

    companies = _extract_companies(client, keyword, market, competitor)
    if not companies:
        print("  [Finance] No listed companies identified, skipping.")
        return {"invest_report": None}

    print(f"  [Finance] Searching financials for {len(companies)} companies...")
    enriched = [_search_company_financials(c) for c in companies]

    print("  [Finance] Generating investment report...")

    company_sections = []
    for c in enriched:
        company_sections.append(
            f"### {c['name']} ({c.get('code', 'N/A')} - {c.get('exchange', 'N/A')})\n"
            f"Role: {c.get('role', 'N/A')}\n"
            f"Financial Data: {c.get('financial_snippets', 'N/A')}"
        )

    prompt = f"""You are a senior equity research analyst. Based on the supply chain research for "{keyword}", write a comprehensive investment analysis report in bilingual (Chinese/English) Markdown format.

## Companies Identified in the Supply Chain

{chr(10).join(company_sections)}


## Report Template

# {keyword} 产业链投资分析报告 / Supply Chain Investment Analysis Report

## 产业链概览 / Supply Chain Overview
[Industry value chain structure: upstream/midstream/downstream, key nodes]

## 核心标的分析 / Core Stock Analysis
[For each company above, provide:
- Company profile and competitive moat
- Recent financial performance (revenue, profit, growth rates)
- Valuation assessment (P/E, relative to industry)
- Key catalysts (new products, policy tailwinds, capacity expansion)
- Risk factors (customer concentration, technology risk, regulation)]

## 估值对比 / Valuation Comparison
[Table: company | market cap | revenue | net profit | P/E | recommendation level]

## 投资策略 / Investment Strategy
[Short-term (3-6 months), medium-term (6-12 months), long-term (1-3 years) allocation suggestions]

## 风险提示 / Risk Notes
[Macro risks, industry risks, company-specific risks]

## 数据来源 / Data Sources

Write directly in Markdown. No preamble. Each section in both Chinese and English.
If financial data is insufficient for a company, clearly note "Insufficient data — requires further research"."""

    invest_report = _call_deepseek(client, prompt)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    safe_keyword = re.sub(r"[^\w\-]", "_", keyword)
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "output",
    )
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{safe_keyword}_投资分析_{timestamp}.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(invest_report)

    return {"invest_report": filepath}
