import os
import re
import time
from datetime import datetime
from typing import TypedDict, Optional

from openai import (
    OpenAI,
    APIError,
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
    InternalServerError,
)
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from agents.market import run_market_agent
from agents.competitor import run_competitor_agent
from agents.discover import run_discover_agent


class AgentState(TypedDict):
    keyword: str
    market_result: Optional[dict]
    competitor_result: Optional[dict]
    final_report: Optional[str]


def prepare_node(state: AgentState) -> dict:
    print(f"\n[Orchestrator] Starting research for: {state['keyword']}")
    return {"keyword": state["keyword"]}


def fanout_to_agents(state: AgentState) -> list[Send]:
    return [
        Send("market_agent", {"keyword": state["keyword"]}),
        Send("competitor_agent", {"keyword": state["keyword"]}),
    ]


def market_agent_node(state: AgentState) -> dict:
    print("[MarketAgent] Searching market data...")
    return run_market_agent(state)


def competitor_agent_node(state: AgentState) -> dict:
    print("[CompetitorAgent] Searching competitor data...")
    return run_competitor_agent(state)


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
            raise RuntimeError(
                f"DeepSeek API error after retry: {e}"
            ) from e


def synthesize_node(state: AgentState) -> dict:
    print("[Synthesize] Generating report with DeepSeek...")

    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )

    keyword = state["keyword"]
    market = state.get("market_result") or {}
    competitor = state.get("competitor_result") or {}

    prompt = f"""You are an e-commerce product selection analyst. Based on the research data below, write a comprehensive, bilingual (Chinese/English) product selection report in Markdown format.

Category: {keyword}

## Market Research Data
{market.get('overview', 'No data available')}

## Competitor Analysis Data
{competitor.get('competitors', 'No data available')}

## Important Instructions
- If any section has no or insufficient data, write "Insufficient data available for this section" rather than fabricating information.
- Write each section in both Chinese and English.

## Report Template
Please follow this exact structure:

# {keyword} 选品调研报告 / Product Selection Research Report

## 调研概览 / Research Overview
[Summarize the category and key findings]

## 市场规模与趋势 / Market Size & Trends
[Market size data, growth rates, emerging trends]

## 竞品格局 / Competitive Landscape
[Major competitors, their positioning, market share if available]

## 价格区间分析 / Pricing Analysis
[Price ranges across competitors, budget/premium segments]

## 选品建议 / Product Selection Recommendations
[Actionable recommendations: what types of products to select, what niches exist, what to avoid]

## 风险提示 / Risk Notes
[Market risks, saturation warnings, regulatory concerns]

## 数据来源 / Data Sources
[List all data sources used]

Write directly in Markdown. No preamble, no "here is the report" — output the report directly."""

    report = _call_deepseek(client, prompt)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    safe_keyword = re.sub(r"[^\w\-]", "_", keyword)
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "output",
    )
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{safe_keyword}_{timestamp}.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)

    return {"final_report": filepath}


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("prepare", prepare_node)
    builder.add_node("market_agent", market_agent_node)
    builder.add_node("competitor_agent", competitor_agent_node)
    builder.add_node("synthesize", synthesize_node)

    builder.add_edge(START, "prepare")
    builder.add_conditional_edges("prepare", fanout_to_agents)
    builder.add_edge("market_agent", "synthesize")
    builder.add_edge("competitor_agent", "synthesize")
    builder.add_edge("synthesize", END)

    return builder.compile()


# ── Discover mode: cross-category Top 5 ──


class DiscoverState(TypedDict):
    categories: Optional[list[dict]]
    final_report: Optional[str]


def discover_synthesize_node(state: DiscoverState) -> dict:
    print("[Synthesize] Scoring categories and generating Top 5 report...")

    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )

    categories = state.get("categories") or []

    # Build compact per-category summary for the prompt
    cat_summaries = []
    for i, cat in enumerate(categories, 1):
        m = cat.get("market", {})
        c = cat.get("competitor", {})
        cat_summaries.append(
            f"### Category {i}: {cat['name']}\n"
            f"Market: {m.get('overview', 'N/A')[:500]}\n"
            f"Competitors: {c.get('competitors', 'N/A')[:500]}"
        )

    prompt = f"""You are a senior e-commerce investment analyst. You have research data for {len(categories)} product categories. Score each category on 5 dimensions (1-10, 10=best), rank them, and select the TOP 5 for trade recommendation.

## Scoring Dimensions
1. **Market Size (市场规模)**: Current market size — bigger = higher score
2. **Growth Rate (增长率)**: Year-over-year growth — faster = higher score
3. **Low Competition (竞争度)**: Lower competition intensity = higher score (less saturated)
4. **Profit Margin (利润空间)**: Typical margin potential — higher = higher score
5. **Low Entry Barrier (入局门槛)**: Lower barrier = higher score (easier to enter)

## Category Research Data

{chr(10).join(cat_summaries)}

## Output Template

Write a comprehensive, bilingual (Chinese/English) Markdown report:

# 跨境电商选品 Top 5 推荐报告 / E-Commerce Product Selection Top 5 Report

## 评分总览 / Scoring Overview
[Scoring table — all {len(categories)} categories × 5 dimensions + total score]

## Top 1: [Category Name] / [品类名]
### 推荐理由 / Why Recommended
### 市场数据 / Market Data
### 竞品格局 / Competitive Landscape
### 选品方向 / Product Direction
### 风险提示 / Risk Notes

## Top 2: [Category Name] / [品类名]
[...same structure...]

## Top 3-5
[Same structure for each]

## 综合建议 / Overall Recommendation
[Which categories to prioritize based on different seller profiles]

## 数据来源 / Data Sources

Write directly in Markdown. No preamble, no "here is the report"."""

    report = _call_deepseek(client, prompt)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "output",
    )
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"Top5_选品推荐_{timestamp}.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)

    return {"final_report": filepath}


def build_discover_graph():
    builder = StateGraph(DiscoverState)

    builder.add_node("discover_scan", run_discover_agent)
    builder.add_node("synthesize", discover_synthesize_node)

    builder.add_edge(START, "discover_scan")
    builder.add_edge("discover_scan", "synthesize")
    builder.add_edge("synthesize", END)

    return builder.compile()
