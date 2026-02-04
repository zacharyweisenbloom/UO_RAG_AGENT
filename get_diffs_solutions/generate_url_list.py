import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
import time
import re
from collections import deque
import argparse
import hashlib
from thread_solution import compute_page_hash

base_url = "https://uoregon.edu"

def crawl_uoregon(start_url="https://uoregon.edu", delay=0.0, max_pages=5000000000000000):
    visited = set()
    to_visit = deque([start_url])   
    request_headers = {"Accept": "text/html"}
    session = requests.Session()
    session.headers.update(request_headers)
    with open("uoregon_urls_test.xml", "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n')
        try:
            while to_visit and len(visited) < max_pages:
                current_url = to_visit.popleft()
                if current_url in visited:
                        continue
                try:
                    print(f"Crawling: {current_url}") 
                    response = session.get(current_url, timeout=10)
                    last_mod = response.headers.get("Last-Modified")
                    headers = response.headers
                    if response.status_code != 200:
                        print(f"Failed to retrieve {current_url}: {response.status_code}")
                        continue
                    visited.add(current_url)
                    page_hash = compute_page_hash(current_url)
                    f.write(f"  <url><loc>{current_url}</loc><hash>{page_hash}</hash><lastmod>{last_mod}</lastmod></url>\n")
                    f.flush()
                    soup = BeautifulSoup(response.content, 'lxml')
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        full_url = urljoin(current_url, href)
                        norm_url = normalize_url(full_url)  # Normalize by removing fragment
                        
                        # Check if the URL is valid and belongs to uoregon.edu
                        if is_valid_url(norm_url) and (norm_url not in visited) and (norm_url not in to_visit):
                            to_visit.append(norm_url)

                    #time.sleep(delay)  # Respectful crawling delay 
                except Exception as e:
                    print(f"Error crawling {current_url}: {e}")
                    continue
        finally:
            f.write('</urlset>\n')
            print("XML footer appended and file closed.")
            return visited  # return partial results








def is_valid_url(url):
    parsed = urlparse(url)
    REPEATED_SEGMENT = re.compile(r'(?:^|/)([^/]+)/\1(/|$)')
    return (
        parsed.scheme in {"http", "https"}
        and parsed.hostname  # guards against mailto:, javascript:, etc.
        and parsed.hostname.endswith("uoregon.edu")
        and parsed.hostname != "pages.uoregon.edu"
        and not any(parsed.path.endswith(ext) for ext in [".pdf", ".xml", ".jpg", ".jpeg", ".png", ".gif"])
        and not re.search(REPEATED_SEGMENT, parsed.path)  # Avoid repeated segments like /foo/bar/foo
        and not has_repeated_subpath(url)
    )

def normalize_url(url):
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))

def has_repeated_subpath(url):
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split('/') if p]
    n = len(parts)

    BASE = 257
    MOD = 10**9 + 7

    # Precompute prefix hashes and powers
    hashes = [0] * (n + 1)
    power = [1] * (n + 1)
    for i in range(n):
        hashes[i+1] = (hashes[i] * BASE + hash(parts[i])) % MOD
        power[i+1] = (power[i] * BASE) % MOD

    seen = set()
    for length in range(1, n + 1):  # check all lengths
        for i in range(n - length + 1):
            j = i + length
            h = (hashes[j] - hashes[i] * power[length]) % MOD
            if h in seen:
                return True
            seen.add(h)

    return False

def main():
    parser = argparse.ArgumentParser(description="Crawl University of Oregon website and save URLs.")

    print("Starting crawl...")
    urls = crawl_uoregon()
    print(f"Crawled {len(urls)} pages.")
    
    print("Crawl complete. URLs saved to uoregon_urls.txt.")

if __name__ == "__main__": 
    main()

