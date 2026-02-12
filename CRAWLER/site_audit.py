#!/usr/bin/env python3
"""
site_audit.py — Crawl a website, find broken internal links, and generate an HTML report.
No external dependencies. Uses urllib + html.parser + concurrent.futures.

Example:
  python site_audit.py https://example.com --max-pages 200 --workers 12 --out report.html
"""

import argparse
import concurrent.futures as cf
import html
import os
import re
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urldefrag, urlparse
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


# ----------------------------- HTML parsing -----------------------------

class LinkAndTitleParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links: List[str] = []
        self._in_title = False
        self._title_chunks: List[str] = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        attrs_dict = dict((k.lower(), v) for k, v in attrs if k and v is not None)

        # Links we care about: <a href="...">
        if tag == "a":
            href = attrs_dict.get("href")
            if href:
                self.links.append(href)

        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag.lower() == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self._title_chunks.append(data)

    @property
    def title(self) -> str:
        return " ".join(" ".join(self._title_chunks).split()).strip()


# ----------------------------- Core types -----------------------------

@dataclass
class FetchResult:
    url: str
    status: int
    content_type: str
    elapsed_ms: int
    title: str
    links: List[str]
    error: str = ""


# ----------------------------- Utilities -----------------------------

def normalize_url(base_url: str, raw_href: str) -> Optional[str]:
    """
    Convert a raw href into an absolute URL and remove fragment (#...).
    Skip mailto:, tel:, javascript:, etc.
    """
    raw_href = raw_href.strip()

    if not raw_href:
        return None

    lowered = raw_href.lower()
    if lowered.startswith(("mailto:", "tel:", "javascript:", "data:")):
        return None

    # Join relative to base
    abs_url = urljoin(base_url, raw_href)

    # Drop fragments
    abs_url, _frag = urldefrag(abs_url)

    # Normalize: remove trailing slash except root
    parsed = urlparse(abs_url)
    if parsed.scheme not in ("http", "https"):
        return None

    # Collapse duplicate slashes in path (safe-ish)
    path = re.sub(r"/{2,}", "/", parsed.path or "/")

    # Remove trailing slash if not root
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    rebuilt = parsed._replace(path=path).geturl()
    return rebuilt


def same_site(a: str, b: str) -> bool:
    pa, pb = urlparse(a), urlparse(b)
    return pa.scheme == pb.scheme and pa.netloc == pb.netloc


def is_html_content(content_type: str) -> bool:
    ct = (content_type or "").lower()
    return "text/html" in ct or "application/xhtml+xml" in ct


def fetch(url: str, timeout: float = 10.0, user_agent: str = "site-audit/1.0") -> FetchResult:
    start = time.time()
    try:
        req = Request(url, headers={"User-Agent": user_agent, "Accept": "text/html,application/xhtml+xml,*/*"})
        with urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", 200)
            headers = resp.headers
            content_type = headers.get("Content-Type", "") or ""
            raw = resp.read(2_000_000)  # cap to 2MB

        elapsed_ms = int((time.time() - start) * 1000)

        title = ""
        links: List[str] = []
        if is_html_content(content_type):
            # Try decoding
            text = raw.decode("utf-8", errors="replace")
            parser = LinkAndTitleParser()
            parser.feed(text)
            title = parser.title
            links = parser.links

        return FetchResult(
            url=url,
            status=int(status),
            content_type=content_type.split(";")[0].strip(),
            elapsed_ms=elapsed_ms,
            title=title,
            links=links,
        )

    except HTTPError as e:
        elapsed_ms = int((time.time() - start) * 1000)
        return FetchResult(url=url, status=int(e.code), content_type="", elapsed_ms=elapsed_ms, title="", links=[], error=str(e))
    except URLError as e:
        elapsed_ms = int((time.time() - start) * 1000)
        return FetchResult(url=url, status=0, content_type="", elapsed_ms=elapsed_ms, title="", links=[], error=str(e.reason))
    except Exception as e:
        elapsed_ms = int((time.time() - start) * 1000)
        return FetchResult(url=url, status=0, content_type="", elapsed_ms=elapsed_ms, title="", links=[], error=repr(e))


# ----------------------------- Crawl + audit -----------------------------

def crawl(start_url: str, max_pages: int, workers: int, timeout: float) -> Tuple[Dict[str, FetchResult], Dict[str, Set[str]]]:
    """
    Returns:
      results[url] = FetchResult
      edges[from_url] = set(to_url)  (internal links only)
    """
    start_url = normalize_url(start_url, start_url) or start_url
    if not start_url.startswith(("http://", "https://")):
        raise ValueError("start_url must be http(s)")

    seen: Set[str] = set()
    results: Dict[str, FetchResult] = {}
    edges: Dict[str, Set[str]] = defaultdict(set)

    q = deque([start_url])
    seen.add(start_url)

    # We use a rolling pool: submit a batch, as pages return add new links.
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        in_flight: Dict[cf.Future, str] = {}

        def submit_one(u: str):
            fut = pool.submit(fetch, u, timeout)
            in_flight[fut] = u

        # Prime
        while q and len(results) < max_pages and len(in_flight) < workers:
            submit_one(q.popleft())

        while in_flight:
            done, _ = cf.wait(in_flight.keys(), return_when=cf.FIRST_COMPLETED)

            for fut in done:
                url = in_flight.pop(fut)
                res = fut.result()
                results[url] = res

                # Only parse links if HTML and successful-ish
                if res.status and 200 <= res.status < 400 and is_html_content(res.content_type):
                    for raw_href in res.links:
                        norm = normalize_url(url, raw_href)
                        if not norm:
                            continue

                        # Only traverse internal pages, but record internal edges
                        if same_site(start_url, norm):
                            edges[url].add(norm)
                            if norm not in seen and len(seen) < max_pages:
                                seen.add(norm)
                                q.append(norm)

                # Keep submitting while we have capacity
                while q and len(results) + len(in_flight) < max_pages and len(in_flight) < workers:
                    submit_one(q.popleft())

    return results, edges


def build_report(start_url: str, results: Dict[str, FetchResult], edges: Dict[str, Set[str]]) -> str:
    # Find broken internal links: edges that point to URL with status >=400 or status==0 or missing
    broken_by_source: Dict[str, List[Tuple[str, str]]] = defaultdict(list)  # source -> [(target, reason)]
    status_counts = defaultdict(int)

    for url, res in results.items():
        key = res.status if res.status else "ERR"
        status_counts[key] += 1

    for src, targets in edges.items():
        for tgt in targets:
            r = results.get(tgt)
            if r is None:
                broken_by_source[src].append((tgt, "Not crawled"))
            elif r.status == 0:
                broken_by_source[src].append((tgt, f"Network error: {r.error}"))
            elif r.status >= 400:
                broken_by_source[src].append((tgt, f"HTTP {r.status}"))
            # else OK

    # Sort pages by status then latency
    pages_sorted = sorted(results.values(), key=lambda r: (0 if 200 <= r.status < 400 else 1, -r.status, r.elapsed_ms))

    total = len(results)
    broken_sources = sorted(broken_by_source.keys(), key=lambda k: len(broken_by_source[k]), reverse=True)
    total_broken = sum(len(v) for v in broken_by_source.values())

    def esc(s: str) -> str:
        return html.escape(s or "")

    # Small embedded CSS for nice look
    css = """
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,sans-serif;margin:24px;max-width:1100px}
    h1{margin:0 0 6px 0}
    .muted{color:#666}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}
    .card{border:1px solid #ddd;border-radius:12px;padding:14px;background:#fff}
    table{width:100%;border-collapse:collapse}
    th,td{padding:8px;border-bottom:1px solid #eee;vertical-align:top}
    th{text-align:left;background:#fafafa;position:sticky;top:0}
    .pill{display:inline-block;padding:2px 8px;border-radius:999px;border:1px solid #ddd;font-size:12px}
    .ok{border-color:#bfe7c5;background:#f3fff5}
    .bad{border-color:#f0b4b4;background:#fff5f5}
    .warn{border-color:#f0d7a7;background:#fffaf0}
    .mono{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;font-size:12px}
    a{color:#0a58ca;text-decoration:none}
    a:hover{text-decoration:underline}
    details{margin:10px 0}
    summary{cursor:pointer}
    """

    def status_badge(s: int) -> str:
        if s == 0:
            cls = "bad"
            txt = "ERR"
        elif 200 <= s < 400:
            cls = "ok"
            txt = str(s)
        elif 400 <= s < 500:
            cls = "warn"
            txt = str(s)
        else:
            cls = "bad"
            txt = str(s)
        return f'<span class="pill {cls}">{txt}</span>'

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    status_rows = "".join(
        f"<tr><td class='mono'>{esc(str(k))}</td><td>{v}</td></tr>"
        for k, v in sorted(status_counts.items(), key=lambda kv: (999 if kv[0] == "ERR" else int(kv[0]),))
    )

    pages_rows = []
    for r in pages_sorted:
        title = r.title or ""
        if len(title) > 80:
            title = title[:77] + "…"
        pages_rows.append(
            "<tr>"
            f"<td>{status_badge(r.status)} <a class='mono' href='{esc(r.url)}'>{esc(r.url)}</a></td>"
            f"<td>{esc(title)}</td>"
            f"<td class='mono'>{esc(r.content_type)}</td>"
            f"<td class='mono'>{r.elapsed_ms} ms</td>"
            "</tr>"
        )

    broken_sections = []
    for src in broken_sources:
        items = broken_by_source[src]
        li = "".join(
            f"<li><span class='mono'>{esc(tgt)}</span> — <span class='muted'>{esc(reason)}</span></li>"
            for tgt, reason in sorted(items, key=lambda x: x[0])
        )
        broken_sections.append(
            f"<details><summary><span class='mono'>{esc(src)}</span> "
            f"<span class='pill bad'>{len(items)} broken</span></summary><ul>{li}</ul></details>"
        )

    broken_html = "".join(broken_sections) if broken_sections else "<p class='muted'>No broken internal links found 🎉</p>"

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Site Audit Report</title>
<style>{css}</style>
</head>
<body>
  <h1>Site Audit Report</h1>
  <div class="muted">Start: <span class="mono">{esc(start_url)}</span> · Generated: {esc(now)}</div>

  <div class="grid">
    <div class="card">
      <h3 style="margin:0 0 8px 0">Summary</h3>
      <div>Total pages crawled: <b>{total}</b></div>
      <div>Broken internal links: <b>{total_broken}</b></div>
      <div class="muted" style="margin-top:8px">Tip: broken links are grouped by source page below.</div>
    </div>
    <div class="card">
      <h3 style="margin:0 0 8px 0">Status codes</h3>
      <table>
        <thead><tr><th>Status</th><th>Count</th></tr></thead>
        <tbody>{status_rows}</tbody>
      </table>
    </div>
  </div>

  <div class="card" style="margin-top:14px">
    <h3 style="margin:0 0 8px 0">Broken internal links</h3>
    {broken_html}
  </div>

  <div class="card" style="margin-top:14px">
    <h3 style="margin:0 0 8px 0">All crawled pages</h3>
    <div class="muted">Sorted by status (OK first) then response time.</div>
    <div style="max-height:520px;overflow:auto;margin-top:10px">
      <table>
        <thead><tr><th>URL</th><th>Title</th><th>Type</th><th>Time</th></tr></thead>
        <tbody>
          {''.join(pages_rows)}
        </tbody>
      </table>
    </div>
  </div>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description="Crawl a website, find broken internal links, and generate an HTML report.")
    ap.add_argument("url", help="Start URL, e.g. https://example.com")
    ap.add_argument("--max-pages", type=int, default=150, help="Max pages to crawl (default: 150)")
    ap.add_argument("--workers", type=int, default=10, help="Concurrent workers (default: 10)")
    ap.add_argument("--timeout", type=float, default=10.0, help="Request timeout seconds (default: 10)")
    ap.add_argument("--out", default="report.html", help="Output HTML report file (default: report.html)")
    args = ap.parse_args()

    start = time.time()
    print(f"[+] Crawling: {args.url}")
    print(f"    max_pages={args.max_pages} workers={args.workers} timeout={args.timeout}s")

    results, edges = crawl(args.url, args.max_pages, args.workers, args.timeout)

    dur = time.time() - start
    print(f"[+] Crawled {len(results)} pages in {dur:.2f}s")

    report = build_report(args.url, results, edges)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)

    abs_path = os.path.abspath(args.out)
    print(f"[+] Report written to: {abs_path}")
    print("    Open it in your browser.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Stopped.")
        sys.exit(1)
