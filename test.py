import requests
from gpt_researcher.scraper.browser.browser import BrowserScraper
from gpt_researcher.scraper.beautiful_soup.beautiful_soup import BeautifulSoupScraper
from gpt_researcher.scraper.web_base_loader.web_base_loader import WebBaseLoaderScraper
from gpt_researcher.scraper import NoDriverScraper
from bs4 import BeautifulSoup
import bs4
import re
from collections import OrderedDict
from gpt_researcher.scraper.utils import get_relevant_images, extract_title


scrapper = BrowserScraper(
    # "https://www.reddit.com/r/bapccanada/comments/1imlftg/rtx_50_series_prices_readjusted_in_canada_little/"
    "https://quotes.toscrape.com/scroll"
)


session = requests.Session()
scrapper2 = BeautifulSoupScraper(
    # "https://www.reddit.com/r/bapccanada/comments/1imlftg/rtx_50_series_prices_readjusted_in_canada_little/",
    "https://quotes.toscrape.com/scroll",
    session=session,
)

scrapper3 = WebBaseLoaderScraper(
    # "https://www.reddit.com/r/bapccanada/comments/1imlftg/rtx_50_series_prices_readjusted_in_canada_little/",
    "https://quotes.toscrape.com/scroll",
    session=session,
)

scrapper4 = NoDriverScraper(
    "https://www.reddit.com/r/bapccanada/comments/1imlftg/rtx_50_series_prices_readjusted_in_canada_little/",
    # "https://quotes.toscrape.com/scroll"
)

text, images, title = scrapper4.scrape()
print(text, "|", len(text), "|", images, "|", title)


with open("test_html.html", "r") as file:
    page_source = file.read()

# page_source = '<p><span class="boldest">Extremely bold</span><p>'
soup = BeautifulSoup(page_source, "html.parser")


def get_text(soup: BeautifulSoup) -> str:
    """Get the relevant text from the soup with improved filtering"""
    text_elements = []
    tags = ["h1", "h2", "h3", "h4", "h5", "p", "li", "div", "span"]

    for element in soup.find_all(tags):
        # Skip empty elements
        if not element.text.strip():
            continue

        # Remove excess whitespace and join lines
        cleaned_text = " ".join(element.text.split())

        # Add the cleaned text to our list of elements
        text_elements.append(cleaned_text)

    # Join all text elements with newlines
    return "\n".join(text_elements)


def clean_soup(soup: BeautifulSoup) -> BeautifulSoup:
    """Clean the soup by removing unwanted tags"""
    for tag in soup.find_all(
        [
            "script",
            "style",
            "footer",
            "header",
            "nav",
            "menu",
            "sidebar",
            "svg",
        ]
    ):
        tag.decompose()

    disallowed_class_set = {"nav", "menu", "sidebar", "footer"}

    # clean tags with certain classes
    def does_tag_have_disallowed_class(elem) -> bool:
        if not isinstance(elem, bs4.Tag):
            return False
        return any(
            cls_name in disallowed_class_set for cls_name in elem.get("class", [])
        )

    for tag in soup.find_all(does_tag_have_disallowed_class):
        tag.decompose()

    return soup


title = extract_title(soup)
print(title)
soup = clean_soup(soup)
text = soup.get_text(strip=True, separator="\n")
# text = get_text(soup)

# Remove excess whitespace
text = re.sub(r"\s{2,}", " ", text)


# lines = (line for line in text.splitlines())
# text = "\n".join(lines)


# lines = (line.strip() for line in text.splitlines())
# chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
# text = "\n".join(chunk for chunk in chunks if chunk)

# print(text)
# print(len(text))

# for img in soup.find_all("img"):
#     print(img)

images = get_relevant_images(
    soup,
    "https://www.reddit.com/r/bapccanada/comments/1imlftg/rtx_50_series_prices_readjusted_in_canada_little/",
)
print(images)
title = extract_title(soup)
print(title)

# for chunk in lines:
#     print("----")
#     print(chunk)
