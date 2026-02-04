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

def compute_page_hash(url):
    request_headers = {"Accept": "text/html"}
    response = requests.get(url, headers=request_headers,timeout=10)
    soup = BeautifulSoup(response.content, "lxml")
    body = soup.body
    for tag in body.find_all(["script", "style", "noscript", "iframe", "input", "form"]):
        tag.decompose()

    # Remove query params from all hrefs
    for a in body.find_all("a", href=True):
        a["href"] = a["href"].split("?")[0]

    # Remove dynamically injected elements
    for div in body.select(".be-ix-link-block"):
        div.decompose()
    clean_text = body.get_text(separator=" ", strip=True)
    #page_hash = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()    
    page_hash = hashlib.sha256(soup.body.get_text().encode("utf-8")).hexdigest()    
    return page_hash


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

def parallel_check(urls):
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
        results = executor.map(process_for_parallel, urls)
        for result in results:
            if result:  # print only non-empty results
                print(result)


def process_for_parallel(url):
    """Compare current page hash with stored hash."""
    url, old_hash = url
    #print(f"Checking: {url}")
    new_hash = compute_page_hash(url)
    if new_hash is None:
        pass
    if old_hash is None:
        print("   [NEW] No previous hash recorded.")
    elif new_hash != old_hash:
        print(f"{url} [CHANGED] Page content is different.")
    else:
        pass
        #print("   [UNCHANGED] No difference detected.")

def main():
    urls = load_urls_from_xml(XML_FILE)
    print(f"Loaded {len(urls)} URLs from {XML_FILE}")
    parallel_check(urls)
if __name__ == "__main__":
    main()


