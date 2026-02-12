# Site Audit Tool (Python Website Crawler)

A lightweight website crawler that scans internal pages, detects broken links, and generates a visual HTML report.

This project is implemented in a **single Python file** using only the **standard library** (no external dependencies).

It demonstrates:
- HTTP requests
- HTML parsing
- URL normalization
- concurrency (ThreadPoolExecutor)
- graph traversal (BFS crawling)
- report generation

---

## Features

- Crawls internal pages of a website
- Detects broken internal links
- Extracts page titles
- Measures response time
- Generates a clean HTML report
- Concurrent crawling for performance
- No external libraries required

---

## Requirements

Python 3.9+

No pip packages needed.

---

## How to Run

Clone or download the project and run:

```bash
python site_audit.py https://example.com
