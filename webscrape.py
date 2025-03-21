"""
This module provides a web scraping utility to extract text content from a website. 
It crawls the website up to a specified number of pages, collects text content, 
and saves the scraped data in JSON format.

Key Features:
- Crawl and scrape text content from a website.
- Limit the number of pages to scrape.
- Save the scraped content to a JSON file in the 'uploads' folder.
- Ensure only valid URLs within the same domain are processed.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urldefrag
import json
import time
from collections import deque
import os

class WebScraper:
    def __init__(self, base_url, max_pages=20, output_folder="uploads"):
        self.base_url = base_url
        self.max_pages = max_pages
        self.output_folder = output_folder
        self.visited_urls = set()
        self.queue = deque([base_url])
        self.scraped_data = []

    def scrape(self):
        """
        Scrapes the website starting from the base URL, collecting text content up to the specified number of pages.
        """
        while self.queue and len(self.visited_urls) < self.max_pages:
            url = self.queue.popleft()
            url, _ = urldefrag(url)

            if url in self.visited_urls:
                continue

            self.visited_urls.add(url)

            try:
                response = requests.get(url, timeout=5)
                response.raise_for_status()
            except requests.RequestException as e:
                print(f"Failed to fetch {url}: {e}")
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            page_content = soup.get_text(separator="\n", strip=True)

            print(f"Scraped: {url} | Content Length: {len(page_content)}")

            self.scraped_data.append({"url": url, "content": page_content})

            for link in soup.find_all("a", href=True):
                full_url = urljoin(self.base_url, link["href"])
                if self.is_valid_url(full_url):
                    self.queue.append(full_url)

            time.sleep(1)

        self.save_content()

    def is_valid_url(self, url):
        """
        Validates a URL to ensure it belongs to the same domain and is not already visited.

        Args:
            url (str): The URL to validate.

        Returns:
            bool: True if the URL is valid, False otherwise.
        """
        url, _ = urldefrag(url)
        parsed_url = urlparse(url)
        parsed_base = urlparse(self.base_url)

        return (
            url not in self.visited_urls and
            parsed_url.netloc == parsed_base.netloc and
            not parsed_url.path.endswith((".jpg", ".png", ".pdf", ".zip"))
        )

    def save_content(self):
        """
        Saves the scraped content to a JSON file in the uploads folder.
        """
        parsed_base = urlparse(self.base_url)
        title = parsed_base.netloc.replace('.', '_') + parsed_base.path.replace('/', '_')
        filename = os.path.join(self.output_folder, f"{title}.json")

        try:
            with open(filename, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except FileNotFoundError:
            existing_data = []

        existing_data.extend(self.scraped_data)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)

        print(f"Saved scraped content to {filename}")

