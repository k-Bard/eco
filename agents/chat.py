"""
Conversational Agent — perception layer that understands user intent and
orchestrates tool calls to research, discover, and invest sub-agents.
Uses DeepSeek Chat (fast, function-calling) to drive the dialogue.
"""
import os
import json
from openai import OpenAI

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "research_product",
            "description": (
                "深度调研一个产品品类，生成选品报告。包括市场规模、增长趋势、竞品格局、"
                "价格区间、选品建议。当用户想了解某个具体品类时调用。"
                "Deep-dive product category research with market size, trends, competitors, pricing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "品类关键词，如 'AI硬件'、'智能手表'",
                    }
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "discover_trending",
            "description": (
                "发现当前热门的电商品类，生成Top 5选品推荐报告。"
                "当用户想找方向、不知道卖什么、或者问'最近什么火'时调用。"
                "Discover trending e-commerce categories, rank Top 5 with scores."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "invest_analysis",
            "description": (
                "对品类的产业链进行投资分析，识别上市公司，提供估值对比和投资策略。"
                "当用户提到'投资'、'股票'、'上市公司'、'产业链分析'时调用。"
                "Supply chain investment analysis: identify listed companies, valuation, strategy."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "品类关键词，如 'AI硬件'",
                    }
                },
                "required": ["keyword"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are 选品助手, an AI e-commerce product selection assistant. You help users:
1. Research specific product categories (market size, trends, competitors)
2. Discover trending categories and get Top 5 recommendations
3. Analyze supply chain investments for product categories

Guidelines:
- When a user asks about a specific product category, use research_product.
- When a user asks about trending/trends or "what to sell", use discover_trending.
- When a user mentions investment/stocks/listed companies, use invest_analysis.
- After a tool runs, tell the user what happened and suggest logical next steps.
  Example: After research, mention if there are listed companies in the report and ask if they want investment analysis.
- Respond conversationally in the user's language (Chinese or English).
- Keep responses concise — the detailed reports are in the generated files.
- If the user just wants to chat or ask general questions, respond directly without calling any tool."""


def _make_client():
    return OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )


def _execute_research(keyword: str) -> str:
    from agents.orchestrator import build_graph

    print(f"\n[Chat -> Research] 正在调研: {keyword}")
    graph = build_graph()
    result = graph.invoke({"keyword": keyword})

    output_path = result.get("final_report", "")
    market = result.get("market_result") or {}
    overview = (market.get("overview", "") or "")[:400]

    competitor = result.get("competitor_result") or {}
    comp_text = (competitor.get("competitors", "") or "")[:1000]
    has_companies = any(
        term in comp_text for term in ["股份", "科技", "集团", "Inc", "Corp", "Ltd"]
    )

    summary = f"选品报告已生成: {output_path}\n\n市场概况: {overview}...\n\n"
    if has_companies:
        summary += "报告中涉及产业链上市公司。如需投资分析，回复'帮我做投资分析'。"
    return summary


def _execute_discover() -> str:
    from agents.orchestrator import build_discover_graph

    print("\n[Chat -> Discover] 正在发现热门品类...")
    graph = build_discover_graph()
    result = graph.invoke({})

    output_path = result.get("final_report", "")
    return (
        f"Top 5 选品推荐报告已生成: {output_path}\n\n"
        "如需深入调研其中某个品类，告诉我品类名即可。"
    )


def _execute_invest(keyword: str) -> str:
    from agents.finance import extract_companies, run_finance_agent
    from agents.orchestrator import build_graph

    print(f"\n[Chat -> Invest] 正在分析: {keyword}")

    graph = build_graph()
    state = graph.invoke({"keyword": keyword})

    client = _make_client()
    companies = extract_companies(
        client,
        keyword,
        state.get("market_result") or {},
        state.get("competitor_result") or {},
    )

    if not companies:
        return (
            f"未在 {keyword} 产业链中识别到上市公司。"
            f"建议先调研品类，回复'帮我调研{keyword}'。"
        )

    preview = f"识别到 {len(companies)} 家上市公司:\n"
    for c in companies:
        preview += (
            f"  - {c['name']} ({c.get('code', 'N/A')}) — {c.get('role', 'N/A')}\n"
        )

    invest_result = run_finance_agent(state)
    invest_path = invest_result.get("invest_report", "")
    if invest_path:
        preview += f"\n投资分析报告已生成: {invest_path}"
    else:
        preview += "\n投资分析报告生成失败。"

    return preview


TOOL_MAP = {
    "research_product": _execute_research,
    "discover_trending": _execute_discover,
    "invest_analysis": _execute_invest,
}


def run_tool(name: str, arguments: dict) -> str:
    func = TOOL_MAP.get(name)
    if not func:
        return f"Unknown tool: {name}"
    try:
        return func(**arguments)
    except Exception as e:
        return f"工具执行出错: {e}"


def start_chat():
    """Main conversational loop."""
    client = _make_client()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("\n" + "=" * 60)
    print("  选品助手 — E-Commerce Product Selection Agent")
    print("  我可以帮你: 调研品类 | 发现热门趋势 | 产业链投资分析")
    print("  输入 'quit' 或 'exit' 退出")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(msg.model_dump())

            for tc in msg.tool_calls:
                func_name = tc.function.name
                func_args = json.loads(tc.function.arguments)

                print(f"\n  -> 调用工具: {func_name}({func_args})")

                result = run_tool(func_name, func_args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
            )
            final_msg = response.choices[0].message.content
            print(f"\n{final_msg}")
            messages.append({"role": "assistant", "content": final_msg})
        else:
            print(f"\n{msg.content}")
            messages.append({"role": "assistant", "content": msg.content})
