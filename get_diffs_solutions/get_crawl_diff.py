import requests
import hashlib
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import concurrent.futures
import aiohttp
import asyncio
MAX_CONCURRENCY=50
XML_FILE = "uoregon_urls_test.xml"
RETRIES=1

async def fetch_and_hash(session, url, old_hash):
    """Fetch URL and compute its page hash (with retries)."""
    for attempt in range(RETRIES):
        try:
            async with session.get(url, timeout=10) as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}")
                html = await response.text()
                soup = BeautifulSoup(html, "lxml")
                body = soup.body
                if not body:
                    return url, None
                text = body.get_text(separator=" ", strip=True)
                page_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

                # Compare to old_hash
                if old_hash is None:
                    return url, "[NEW] No previous hash recorded"
                elif page_hash != old_hash:
                    return url, "[CHANGED] Page content is different"
                else:
                    return url, None
        except Exception as e:
            if attempt == RETRIES - 1:
                return url, f"[ERROR] {e}"
            await asyncio.sleep(1)  # backoff before retry




def load_urls_from_xml(xml_file):
    """Parse XML and return a list of (url, old_hash)."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = []
    for url_node in root.findall("sm:url", ns):
        loc = url_node.find("sm:loc", ns).text
        hash_node = url_node.find("sm:hash", ns)
        old_hash = hash_node.text if hash_node is not None else None
        urls.append((loc, old_hash))
    return urls



async def parallel_check_async(urls):
    """Run async parallel processing with aiohttp."""
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENCY)
    headers = {"Accept": "text/html"}
    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        tasks = [fetch_and_hash(session, url, old_hash) for url, old_hash in urls]
        for future in asyncio.as_completed(tasks):
            url, result = await future
            if result:
                pass
                #print(f"{url} {result}")

def parallel_check(urls):
    """Sync wrapper to call the async version."""
    asyncio.run(parallel_check_async(urls))

def main():
    urls = load_urls_from_xml(XML_FILE)
    print(f"Loaded {len(urls)} URLs from {XML_FILE}")
    #parallel_check(urls) 
    parallel_check(urls)
if __name__ == "__main__":
    main()


