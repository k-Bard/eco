import os
import time
from tavily import TavilyClient

_client = None


def get_tavily_client() -> TavilyClient:
    global _client
    if _client is None:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY not set in environment")
        _client = TavilyClient(api_key=api_key)
    return _client


def search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

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
        except (ConnectionError, TimeoutError, OSError) as e:
            if attempt == 0:
                time.sleep(2)
                continue
            print(f"[Tavily] Search failed for '{query}': {e}")
            return []
        except Exception:
            raise
