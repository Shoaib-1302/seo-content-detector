"""
HTML Parsing and Content Extraction Utilities
"""

import re
import requests
from bs4 import BeautifulSoup
from time import sleep

def scrape_url(url, timeout=10):
    """
    Scrape a single URL and return HTML content.
    
    Args:
        url (str): URL to scrape
        timeout (int): Request timeout in seconds
        
    Returns:
        str: HTML content or empty string on failure
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {str(e)}")
        return ""
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return ""

def parse_html_content(html_content):
    """
    Parse HTML content and extract meaningful text.
    
    Args:
        html_content (str): Raw HTML content
        
    Returns:
        dict: Dictionary with title and body_text
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script, style, and navigation elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()
        
        # Extract title
        title = ""
        if soup.title:
            title = soup.title.string.strip() if soup.title.string else ""
        
        # Try to find main content using common patterns
        body_text = ""
        
        # Priority 1: Look for article or main tags
        main_content = soup.find('article') or soup.find('main')
        if main_content:
            body_text = main_content.get_text()
        else:
            # Priority 2: Look for content divs
            content_divs = soup.find_all(['div'], class_=re.compile(r'content|article|post|entry', re.I))
            if content_divs:
                body_text = ' '.join([div.get_text() for div in content_divs])
            else:
                # Priority 3: Get all paragraphs
                paragraphs = soup.find_all('p')
                if paragraphs:
                    body_text = ' '.join([p.get_text() for p in paragraphs])
                else:
                    # Fallback: Get all text
                    body_text = soup.get_text()
        
        # Clean the text
        body_text = re.sub(r'\s+', ' ', body_text).strip()
        
        return {
            'title': title,
            'body_text': body_text
        }
    except Exception as e:
        print(f"Error parsing HTML: {str(e)}")
        return {
            'title': "",
            'body_text': ""
        }

def scrape_and_parse_url(url, delay=0):
    """
    Convenience function to scrape and parse a URL.
    
    Args:
        url (str): URL to process
        delay (float): Delay after scraping in seconds
        
    Returns:
        tuple: (html_content, parsed_data)
    """
    html_content = scrape_url(url)
    if delay > 0:
        sleep(delay)
    
    parsed_data = parse_html_content(html_content)
    return html_content, parsed_data

def extract_metadata(soup):
    """
    Extract metadata from BeautifulSoup object.
    
    Args:
        soup: BeautifulSoup object
        
    Returns:
        dict: Metadata dictionary
    """
    metadata = {
        'description': '',
        'keywords': '',
        'author': '',
        'og_title': '',
        'og_description': ''
    }
    
    try:
        # Meta description
        desc_tag = soup.find('meta', attrs={'name': 'description'})
        if desc_tag and desc_tag.get('content'):
            metadata['description'] = desc_tag['content']
        
        # Meta keywords
        keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
        if keywords_tag and keywords_tag.get('content'):
            metadata['keywords'] = keywords_tag['content']
        
        # Author
        author_tag = soup.find('meta', attrs={'name': 'author'})
        if author_tag and author_tag.get('content'):
            metadata['author'] = author_tag['content']
        
        # Open Graph tags
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
            metadata['og_title'] = og_title['content']
        
        og_desc = soup.find('meta', property='og:description')
        if og_desc and og_desc.get('content'):
            metadata['og_description'] = og_desc['content']
    
    except Exception as e:
        print(f"Error extracting metadata: {str(e)}")
    
    return metadata