import sys
import os
from dotenv import load_dotenv


def main():
    load_dotenv()

    if "DEEPSEEK_API_KEY" not in os.environ:
        print("Error: DEEPSEEK_API_KEY not set in .env file")
        sys.exit(1)
    if "TAVILY_API_KEY" not in os.environ:
        print("Error: TAVILY_API_KEY not set in .env file")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Usage: python main.py <category-keyword>")
        print('Example: python main.py "智能家居"')
        sys.exit(1)

    keyword = sys.argv[1]

    from agents.orchestrator import build_graph

    graph = build_graph()
    result = graph.invoke({"keyword": keyword})

    output_path = result.get("final_report")
    if output_path:
        print(f"\nReport saved to: {output_path}")
    else:
        print("\nError: Failed to generate report")
        sys.exit(1)


if __name__ == "__main__":
    main()
