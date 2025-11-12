import aiohttp
from typing import Any, Dict
from .base import BaseTool  # if you saved BaseTool in a separate file

class WebSearchTool(BaseTool):
    """Tool for searching the web using the DuckDuckGo Instant Answer API."""
    
    name = "duckduckgo_search"
    description = (
        "Perform a web search using DuckDuckGo and return related results "
        "(abstracts, URLs, and titles)."
    )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on DuckDuckGo."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of search results to return (default 5).",
                    "minimum": 1,
                    "maximum": 20
                }
            },
            "required": ["query"]
        }

    async def run(self, **kwargs) -> Any:
        query = kwargs.get("query")
        max_results = kwargs.get("max_results", 5)
        url = "https://api.duckduckgo.com/"

        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()

        results = []

        # Combine 'RelatedTopics' and 'Results' fields
        related_topics = data.get("RelatedTopics", []) + data.get("Results", [])
        for item in related_topics:
            # Some items are nested dicts
            if "Text" in item and "FirstURL" in item:
                results.append({
                    "title": item.get("Text", "").strip(),
                    "link": item.get("FirstURL", ""),
                    "snippet": item.get("Text", "")
                })
            elif "Topics" in item:
                for subitem in item["Topics"]:
                    if "Text" in subitem and "FirstURL" in subitem:
                        results.append({
                            "title": subitem.get("Text", "").strip(),
                            "link": subitem.get("FirstURL", ""),
                            "snippet": subitem.get("Text", "")
                        })
            if len(results) >= max_results:
                break

        return {
            "query": query,
            "results": results[:max_results]
        }
