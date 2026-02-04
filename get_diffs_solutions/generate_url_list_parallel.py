import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
import time
import re
from queue import Queue
import argparse
import threading
from html import escape
import hashlib

base_url = "https://uoregon.edu"
BLOCKLIST_SUBSTRINGS = ["oregonnews.uoregon.edu/lccn"]

output_file = "uoregon_urls_with_oregonnews.xml"
def is_blocked_by_substrings(url: str) -> bool:
    u = url.lower()
    return any(s in u for s in BLOCKLIST_SUBSTRINGS)



def compute_page_hash(content):

    soup = BeautifulSoup(content, "lxml")
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

# Precompile once (faster + thread-safe to reuse)
REPEATED_SEGMENT = re.compile(r'(?:^|/)([^/]+)/\1(?:/|$)')

def crawl_uoregon(start_url="https://uoregon.edu", delay=0.0, max_pages=5_000_000_000_000_000, workers=16):
    visited = set()         # URLs successfully fetched
    enqueued = set()        # URLs currently queued (prevents dup work)
    visited_lock = threading.Lock()
    enqueued_lock = threading.Lock()
    file_lock = threading.Lock()
    stop_event = threading.Event()

    q = Queue()
    with enqueued_lock:
        q.put(start_url)
        enqueued.add(start_url)

    request_headers = {"Accept": "text/html"}

    def make_session():
        s = requests.Session()
        s.headers.update(request_headers)
        return s

    def worker(file_handle):
        session = make_session()
        while True:
            url = q.get()
            if url is None:
                q.task_done()
                break
            if stop_event.is_set():
                # Drain quickly if we’ve hit the max_pages limit
                q.task_done()
                continue

            try:
                print(f"Crawling: {url}")
                resp = session.get(url, timeout=10)
            except Exception as e:
                print(f"Error crawling {url}: {e}")
                q.task_done()
                continue

            if resp.status_code != 200:
                print(f"Failed to retrieve {url}: {resp.status_code}")
                q.task_done()
                continue

            # Sanity check: only parse HTML
            ctype = resp.headers.get("Content-Type", "")
            if "html" not in ctype.lower():
                q.task_done()
                continue

            # Mark visited only after a successful fetch
            with visited_lock:
                if url in visited:
                    q.task_done()
                    continue
                visited.add(url)
                if len(visited) >= max_pages:
                    stop_event.set()

            last_mod = resp.headers.get("Last-Modified", "")

            # If compute_page_hash refetches, you could replace it with a hash of resp.content
            # to avoid a second network call. Keeping your behavior for now:
            try:
                page_hash = compute_page_hash(resp.content)
            except Exception as e:
                print(f"compute_page_hash failed for {url}: {e}")
                page_hash = ""

            # Thread-safe XML write
            with file_lock:
                file_handle.write(
                    f'  <url><loc>{escape(url)}</loc><hash>{escape(str(page_hash))}</hash><lastmod>{escape(str(last_mod))}</lastmod></url>\n'
                )
                file_handle.flush()

            soup = BeautifulSoup(resp.content, "lxml")
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(url, href)
                norm_url = normalize_url(full_url)

                if not is_valid_url(norm_url):
                    continue

                with enqueued_lock:
                    if (norm_url not in enqueued) and (norm_url not in visited) and not stop_event.is_set():
                        enqueued.add(norm_url)
                        q.put(norm_url)

            if delay:
                time.sleep(delay)

            q.task_done()

    with open(output_file, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n')

        threads = [threading.Thread(target=worker, args=(f,), daemon=True) for _ in range(workers)]
        for t in threads:
            t.start()

        # Wait for all queued work to finish
        q.join()

        # Stop workers cleanly
        for _ in threads:
            q.put(None)
        for t in threads:
            t.join()

        f.write('</urlset>\n')
        print("XML footer appended and file closed.")
        return visited  # partial or full, depending on limits

def is_valid_url(url):
    parsed = urlparse(url)
    return (
        parsed.scheme in {"http", "https"}
        and parsed.hostname
        and parsed.hostname.endswith("uoregon.edu")
        #and parsed.hostname != "pages.uoregon.edu"
        and not any(parsed.path.endswith(ext) for ext in [".pdf", ".xml", ".jpg", ".jpeg", ".png", ".gif"])
        and not REPEATED_SEGMENT.search(parsed.path)  # Avoid repeated single segments like /foo/bar/foo
        and not has_repeated_subpath(url)             # Avoid repeated multi-segment subpaths
        #and not is_blocked_by_substrings(url)
    )

def normalize_url(url):
    parsed = urlparse(url)
    # Strip query + fragment to keep your original normalization
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))

def has_repeated_subpath(url):
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split('/') if p]
    n = len(parts)

    BASE = 257
    MOD = 10**9 + 7

    hashes = [0] * (n + 1)
    power = [1] * (n + 1)
    for i in range(n):
        hashes[i+1] = (hashes[i] * BASE + hash(parts[i])) % MOD
        power[i+1] = (power[i] * BASE) % MOD

    seen = set()
    for length in range(1, n + 1):
        for i in range(n - length + 1):
            j = i + length
            h = (hashes[j] - hashes[i] * power[length]) % MOD
            if h in seen:
                return True
            seen.add(h)
    return False

def main():
    parser = argparse.ArgumentParser(description="Crawl University of Oregon website and save URLs.")
    parser.add_argument("--start-url", default="https://uoregon.edu")
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--max-pages", type=int, default=5_000_000)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    print("Starting crawl...")
    urls = crawl_uoregon(start_url=args.start_url, delay=args.delay, max_pages=args.max_pages, workers=args.workers)
    print(f"Crawled {len(urls)} pages.")
    print("Crawl complete. URLs saved to uoregon_urls_test.xml.")

if __name__ == "__main__":
    main()

