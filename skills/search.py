import os
import time
from tavily import TavilyClient


def get_tavily_client():
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY not set in environment")
    return TavilyClient(api_key=api_key)


def search(query: str, max_results: int = 5) -> list[dict]:
    client = get_tavily_client()

    for attempt in range(2):
        try:
            response = client.search(query=query, max_results=max_results)
            results = response.get("results", [])
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                }
                for r in results
            ]
        except Exception as e:
            if attempt == 0:
                time.sleep(2)
                continue
            print(f"[Tavily] Search failed for '{query}': {e}")
            return []

    return []
