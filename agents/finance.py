import os
import re
import time
from datetime import datetime

import akshare as ak
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


def extract_companies(client: OpenAI, keyword: str, market: dict, competitor: dict) -> list[dict]:
    """DeepSeek identifies listed companies from research data."""
    print("  [Finance] Extracting listed companies from research...")

    raw = (
        f"Market overview: {market.get('overview', '')[:1500]}\n\n"
        f"Competitor data: {competitor.get('competitors', '')[:1500]}"
    )

    prompt = f"""From the following e-commerce product category research for "{keyword}", extract up to 8 publicly traded companies (A-share, HK, US-listed) that are key players in this industry's supply chain.

Research data:
{raw}

Return ONLY a JSON array:
{{"name": "公司中文名", "code": "股票代码(6 digits for A-shares)", "exchange": "深交所/上交所/港交所/纳斯达克/NYSE", "role": "产业链角色"}}

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


def _fetch_astock_snapshot(code: str) -> dict:
    """AKShare: stock snapshot — market cap, PE, PB, industry, total shares."""
    try:
        df = ak.stock_individual_info_em(symbol=code)
        data = {}
        for _, row in df.iterrows():
            data[str(row["item"])] = str(row["value"])

        return {
            "market_cap": data.get("总市值", "N/A"),
            "circulating_cap": data.get("流通市值", "N/A"),
            "industry": data.get("行业", "N/A"),
            "total_shares": data.get("总股本", "N/A"),
            "listing_date": data.get("上市时间", "N/A"),
        }
    except Exception as e:
        print(f"    [AKShare] Snapshot failed for {code}: {e}")
        return {}



def _enrich_with_akshare(company: dict) -> dict:
    """Try AKShare for A-shares. Returns enriched company or original on failure."""
    code = company.get("code", "")
    exchange = company.get("exchange", "")

    if exchange not in ("深交所", "上交所", "北交所") or not code:
        return company

    print(f"    [AKShare] Fetching data for {company['name']} ({code})...")
    akshare_data = _fetch_astock_snapshot(code)

    if akshare_data:
        result = dict(company)
        result["akshare_data"] = akshare_data
        result["data_source"] = "AKShare"
        return result

    return company


def _enrich_with_tavily(company: dict) -> dict:
    """Fallback: Tavily search for financial snippets."""
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
    result["data_source"] = "Tavily"
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

    companies = extract_companies(client, keyword, market, competitor)
    if not companies:
        print("  [Finance] No listed companies identified, skipping.")
        return {"invest_report": None}

    print(f"  [Finance] Enriching {len(companies)} companies...")
    enriched = []
    for c in companies:
        result = _enrich_with_akshare(c)
        if "akshare_data" not in result:
            result = _enrich_with_tavily(c)
        enriched.append(result)

    print("  [Finance] Generating investment report...")
    company_sections = []
    for c in enriched:
        header = f"### {c['name']} ({c.get('code', 'N/A')} - {c.get('exchange', 'N/A')})\n"
        header += f"Role: {c.get('role', 'N/A')}\n"
        header += f"Data Source: {c.get('data_source', 'Unknown')}\n"

        if "akshare_data" in c:
            d = c["akshare_data"]
            header += (
                f"- 总市值 / Market Cap: {d.get('market_cap', 'N/A')}\n"
                f"- 流通市值 / Circulating Cap: {d.get('circulating_cap', 'N/A')}\n"
                f"- PE(TTM): {d.get('pe_ttm', 'N/A')} | PB: {d.get('pb', 'N/A')} | PS: {d.get('ps', 'N/A')}\n"
                f"- 行业 / Industry: {d.get('industry', 'N/A')}\n"
                f"- 总股本 / Total Shares: {d.get('total_shares', 'N/A')}\n"
                f"- 上市日期 / Listed: {d.get('listing_date', 'N/A')}\n"
            )
        else:
            header += f"Financial Data: {c.get('financial_snippets', 'N/A')}\n"

        company_sections.append(header)

    prompt = f"""You are a senior equity research analyst. Based on the supply chain research for "{keyword}", write a comprehensive investment analysis report in bilingual (Chinese/English) Markdown format.

## Companies Identified in the Supply Chain

{chr(10).join(company_sections)}


## Report Template

# {keyword} 产业链投资分析报告 / Supply Chain Investment Analysis Report

## 产业链概览 / Supply Chain Overview
[Industry value chain structure: upstream/midstream/downstream, key nodes. How each identified company fits.]

## 核心标的分析 / Core Stock Analysis
[For each company above, provide detailed investment analysis:
- **公司概况与护城河 / Company Profile & Moat**: Business model, competitive advantages
- **财务表现 / Financial Performance**: Revenue, profit, growth rates — use the AKShare/Tavily data provided
- **估值评估 / Valuation Assessment**: PE/PB analysis vs industry peers
- **催化因素 / Key Catalysts**: Near-term growth drivers
- **风险因素 / Risk Factors**: Company-specific risks]

## 估值对比 / Valuation Comparison
[Table: 公司/Company | 代码/Code | 市值/Market Cap | PE | PB | 行业/Industry | 推荐评级/Recommendation]

## 投资策略 / Investment Strategy
- **短期 (3-6月) / Short-term**
- **中期 (6-12月) / Medium-term**
- **长期 (1-3年) / Long-term**

## 风险提示 / Risk Notes

## 数据来源 / Data Sources

Write directly in Markdown. No preamble. Each section bilingual.
If specific financial data is unavailable, note "数据不足 / Insufficient data."
Include disclaimer: "本报告仅供参考，不构成投资建议."""

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
