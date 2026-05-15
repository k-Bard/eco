import sys
import os
from dotenv import load_dotenv
from openai import OpenAI


def _prompt_invest(keyword: str, result: dict) -> None:
    """Extract companies from research and ask user if they want investment analysis."""
    from agents.finance import extract_companies

    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )

    print("\n[Invest] 正在识别报告中涉及的上市公司...")
    companies = extract_companies(
        client,
        keyword,
        result.get("market_result", {}),
        result.get("competitor_result", {}),
    )

    if not companies:
        print("[Invest] 未在报告中识别到上市公司，跳过投资分析。")
        return

    print(f"\n报告中涉及 {len(companies)} 家上市公司:")
    for c in companies:
        print(f"  - {c['name']} ({c.get('code', 'N/A')}) — {c.get('role', 'N/A')}")

    try:
        choice = input("\n是否生成产业链投资分析报告？(y/n): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    if choice != "y":
        print("[Invest] 已跳过投资分析。")
        return

    print()
    from agents.finance import run_finance_agent

    invest_result = run_finance_agent(result)
    invest_path = invest_result.get("invest_report")
    if invest_path:
        print(f"[投资分析] {invest_path}")
    else:
        print("[Invest] 投资分析生成失败。")


def main():
    load_dotenv()

    if "DEEPSEEK_API_KEY" not in os.environ:
        print("Error: DEEPSEEK_API_KEY not set in .env file")
        sys.exit(1)
    if "TAVILY_API_KEY" not in os.environ:
        print("Error: TAVILY_API_KEY not set in .env file")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py <category-keyword>           Single category deep-dive")
        print("  python main.py <category-keyword> --invest  Deep-dive + prompt for investment analysis")
        print("  python main.py --discover                   Cross-category Top 5")
        print('Example: python main.py "AI硬件"')
        print('Example: python main.py "AI硬件" --invest')
        print("Example: python main.py --discover")
        sys.exit(1)

    if sys.argv[1] == "--discover":
        from agents.orchestrator import build_discover_graph

        graph = build_discover_graph()
        result = graph.invoke({})
    else:
        keyword = sys.argv[1]
        run_invest = "--invest" in sys.argv

        from agents.orchestrator import build_graph

        graph = build_graph()
        result = graph.invoke({"keyword": keyword})

        output_path = result.get("final_report")
        if output_path:
            print(f"\n[选品报告] {output_path}")
        else:
            print("\nError: Failed to generate report")
            sys.exit(1)

        if run_invest:
            _prompt_invest(keyword, result)
        return

    output_path = result.get("final_report")
    if output_path:
        print(f"\n[选品报告] {output_path}")
    else:
        print("\nError: Failed to generate report")
        sys.exit(1)


if __name__ == "__main__":
    main()
