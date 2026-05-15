import os
import re
import time
from datetime import datetime
from typing import TypedDict, Optional

from openai import OpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from agents.market import run_market_agent
from agents.competitor import run_competitor_agent


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


def synthesize_node(state: AgentState) -> dict:
    print("[Synthesize] Generating report with DeepSeek...")

    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )

    market = state.get("market_result", {})
    competitor = state.get("competitor_result", {})

    prompt = f"""You are an e-commerce product selection analyst. Based on the research data below, write a comprehensive, bilingual (Chinese/English) product selection report in Markdown format.

Category: {state['keyword']}

## Market Research Data
{market.get('overview', 'No data available')}

## Competitor Analysis Data
{competitor.get('competitors', 'No data available')}

## Report Template
Please follow this exact structure. Write each section in both Chinese and English:

# {{Keyword}} 选品调研报告 / Product Selection Research Report

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

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            report = response.choices[0].message.content

            # Save report
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            safe_keyword = re.sub(r"[^\w\-]", "_", state["keyword"])
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "output",
            )
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, f"{safe_keyword}_{timestamp}.md")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(report)

            return {"final_report": filepath}
        except Exception as e:
            if attempt == 0:
                time.sleep(3)
                continue
            raise RuntimeError(f"DeepSeek API error after retry: {e}")


def build_graph() -> StateGraph:
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
