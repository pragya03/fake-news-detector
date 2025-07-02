import tldextract
from known_sources import trusted_sources, untrusted_sources
import requests
from bs4 import BeautifulSoup

def get_domain(url):
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}"

def check_source_reliability(url):
    domain = get_domain(url)
    if domain in trusted_sources:
        return "Trusted Source ✅"
    elif domain in untrusted_sources:
        return "Untrusted Source ⚠️"
    else:
        return "Unknown Source ❓"

def get_page_title(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.content, "html.parser")
        title = soup.title.string.strip() if soup.title else "No title found"
        return title
    except Exception as e:
        return f"Error fetching title: {e}"

def google_search_news(query, api_key, cse_id, num_results=3):
    try:
        import urllib.parse
        query_encoded = urllib.parse.quote(query)
        url = f"https://www.googleapis.com/customsearch/v1?q={query_encoded}&key={api_key}&cx={cse_id}&num={num_results}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return [(item["title"], item["link"]) for item in data.get("items", [])]
    except Exception as e:
        return [("Error fetching results", str(e))]
