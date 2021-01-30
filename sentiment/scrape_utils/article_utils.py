import datetime
import re
import string
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from loguru import logger
from tqdm import tqdm

from sentiment.config.constants import HEADERS
from sentiment.scrape_utils import data_utils as du
from sentiment.scrape_utils import motley_utils as mu

YAHOO_FIELDS = ("content", "author", "date", "length")
MOTLEY_FIELDS = mu.ARTICLE_FIELDS
BARRONS_FIELDS = ("categories", "author", "date", "title", "content", "tickers")
INVESTORS_FIELDS = ("categories", "authors", "date", "title", "content", "tickers")
MARKETWATCH_FIELDS = ("categories", "author", "date", "title", "content", "tickers")
INVESTOPEDIA_FIELDS = ("categories", "author", "date", "title", "subtitle", "content", "tickers")


def _validate_date(date, fmt):
    """Helper function to validate a string is a datetime parse-able format and returns datetime object or None"""
    try:
        return datetime.datetime.strptime(date, fmt)
    except ValueError:
        return None


def _get_articles_data(links, source_func, **kwargs) -> Dict[str, Dict[str, Any]]:
    data = OrderedDict()
    for link in (pbar := tqdm(links, total=len(links))) :
        pbar.set_description(link)
        data[link] = source_func(link, **kwargs)
        pbar.update(1)
    pbar.close()
    return data


def yahoo(link: str, default: Optional[Any] = None) -> Dict[str, Any]:
    res = requests.get(link)
    data = OrderedDict((field, default) for field in YAHOO_FIELDS)
    try:
        res.raise_for_status()
    except requests.exceptions.HTTPError as err:
        logger.error(f"HTTP request failed for yahoo article: {link}\nDoes this article exist?\nError:\n{err}")
        return data
    soup = BeautifulSoup(res.text, "html.parser")

    if body := soup.find("div", attrs={"class": "caas-body"}):
        # Removes 32 length preset image strings ie "abc123...456def.png"
        data["content"] = re.sub("[a-z0-9]{32}\.[a-z]{3}", "", body.text)
    else:
        logger.warning(f"Failed to find content div for yahoo article: {link}")

    if article_meta := soup.find("div", attrs={"class": "caas-attr-meta"}):
        if date := _validate_date(article_meta.text, "%B %d, %Y, %I:%M %p"):
            data["date"] = date
        elif meta := re.match(
            r"^(?P<author>.+)?(?P<month>[A-Z][a-z]+) (?P<day>\d{1,2}), (?P<year>\d{4}), "
            r"(?P<hour>\d{1,2}):(?P<minute>\d{1,2}) (?P<period>AM|PM)(:?.(?P<length>\d+) min read)?$",
            article_meta.text,
        ):
            if (author := meta.group("author")) is not None:
                data["author"] = string.capwords(author.strip())
            else:
                logger.warning(f"Failed to parse data for yahoo article: {link}")
            date = f"{meta.group('month')} {meta.group('day')} {meta.group('year')} {meta.group('hour')} {meta.group('minute')} {meta.group('period')}"
            if date := _validate_date(date, "%B %d %Y %I %M %p"):
                data["date"] = date
            else:
                logger.warning(f"Failed to parse date for yahoo article: {link}")
            if length := meta.group("length"):
                data["length"] = int(length)
            else:
                logger.warning(f"Failed to find length for yahoo article: {link}")
        else:
            logger.warning(f"Failed to parse meta data for yahoo article: {link}")
    else:
        logger.warning(f"Failed to find meta div for yahoo article: {link}")
    return data


def yahoo_articles(links: List[str], default: Optional[Any] = None) -> Dict[str, Dict[str, Any]]:
    return _get_articles_data(links, yahoo, default=default)


def motley(link: str, default: Optional[Any] = None) -> Dict[str, Any]:
    sublink = link.split("fool.com")[-1]
    return mu.get_article_data(sublink, default=default)


def motley_articles(links: List[str], default: Optional[Any] = None) -> Dict[str, Dict[str, Any]]:
    return _get_articles_data(links, motley, default=default)


def barrons(link: str, default: Optional[Any] = None):
    res = requests.get(link, headers=HEADERS)
    data = OrderedDict((field, default) for field in BARRONS_FIELDS)
    try:
        res.raise_for_status()
    except requests.exceptions.HTTPError as err:
        logger.error(f"HTTP request failed for barrons article: {link}\nError:\n{err}")
        return data
    soup = BeautifulSoup(res.text, "html.parser")

    if article_categories := soup.find("ul", attrs={"class": "article__flashline"}):
        data["categories"] = [category.text.strip() for category in article_categories.find_all("li")]
    else:
        logger.warning(f"Failed to find article categories for barrons article: {link}")

    if title := soup.find("h1", attrs={"class": "article__headline"}):
        data["title"] = string.capwords(title.text)
    else:
        logger.warning(f"Failed to find title for barrons article: {link}")

    if author := soup.find("span", attrs={"class": "name"}):
        data["author"] = string.capwords(author.text)
    elif author := soup.find("div", attrs={"class": "byline article__byline"}):
        data["author"] = string.capwords(author.text)
    else:
        logger.warning(f"Failed to find author for barrons article: {link}")

    if date := soup.find("time", attrs={"class": "timestamp"}):
        date_text = date.text.strip().replace("Sept.", "Sep.")
        if date_ := _validate_date(date_text, "%b. %d, %Y %I:%M %p ET"):
            data["date"] = date_
        elif date_ := _validate_date(date_text, "%B %d, %Y %I:%M %p ET"):
            data["date"] = date_
        elif "/" in date_text:
            updated_date = date_text.split(" / ")[0]
            if date_ := _validate_date(updated_date, "Updated %b. %d, %Y %I:%M %p ET"):
                data["date"] = date_
            elif date_ := _validate_date(updated_date, "Updated %B %d, %Y %I:%M %p ET"):
                data["date"] = date_
            elif date_ := _validate_date(updated_date, "Updated %B %d, %Y"):
                data["date"] = date_
            else:
                logger.warning(f"Failed to parse updated date for barrons article: {link}\nDate: {updated_date}")
        else:
            logger.warning(f"Failed to parse date for barrons article: {link}\nDate: {date_text}")
    else:
        logger.warning(f"Failed to find date for barrons article: {link}")

    if content := soup.find("div", attrs={"id": "js-article__body"}):
        if content_paragraphs := content.find_all("p"):
            data["content"] = "\n".join(du.get_paragraph(p) for p in content_paragraphs).strip()
        else:
            logger.warning(f"Failed to find content paragraph for barrons article: {link}")
    else:
        logger.warning(f"Failed to find content block for barrons article: {link}")

    if data["content"]:
        data["tickers"] = set(re.findall(r"\(ticker: ([A-Z]+)\)", data["content"]))
    else:
        data["tickers"] = set()

    return data


def barrons_articles(links: List[str], default: Optional[Any] = None) -> Dict[str, Dict[str, Any]]:
    return _get_articles_data(links, barrons, default=default)


def investors(link: str, default: Optional[Any] = None) -> Dict[str, Any]:
    # ! Investors.com does not like web scraping so this will not work with repeated requests in a short period of time
    res = requests.get(link, headers=HEADERS)
    data = OrderedDict((field, default) for field in INVESTORS_FIELDS)
    try:
        res.raise_for_status()
    except requests.exceptions.HTTPError as err:
        logger.error(f"HTTP request failed for investors article: {link}\nError:\n{err}")
        return data
    soup = BeautifulSoup(res.text, "html.parser")

    if article_categories := soup.find("ul", attrs={"class": "post-categories"}):
        data["categories"] = [category.text.strip() for category in article_categories.find_all("li")]
    else:
        logger.warning(f"Failed to find article categories for investors article: {link}")

    if article_meta := soup.find("ul", attrs={"class": "post-meta"}):
        if authors := article_meta.find_all("a", attrs={"rel": "author"}):
            data["authors"] = {string.capwords(author.text) for author in authors}
        else:
            logger.warning(f"Failed to find authors for investors article: {link}")

        if date := article_meta.find("li", attrs={"class": "post-time"}):
            if date_obj := _validate_date(date.text.strip(), "%I:%M %p ET %m/%d/%Y"):
                data["date"] = date_obj
            else:
                logger.warning(f"Failed to parse date for investors article: {link}")
        else:
            logger.warning(f"Failed to find date for investors article: {link}")
    else:
        logger.warning(f"Failed to find article meta for investors article: {link}")

    if title := soup.find("h1", attrs={"class": "header1"}):
        data["title"] = string.capwords(title.text)
    else:
        logger.warning(f"Failed to find title for investors article: {link}")

    if content := soup.find("div", attrs={"class": re.compile(".*post-content.*")}):
        if content_paragraphs := content.find_all("p"):
            data["content"] = "\n".join(du.get_paragraph(p) for p in content_paragraphs).strip()
        else:
            logger.warning(f"Failed to find content paragraph for investors article: {link}")
    else:
        logger.warning(f"Failed to find content block for investors article: {link}")

    if tickers := soup.find_all("a", attrs={"class": "ticker"}):
        data["tickers"] = {ticker.text.upper() for ticker in tickers}
    else:
        data["tickers"] = {}

    return data


def investors_articles(links: List[str], default: Optional[Any] = None) -> Dict[str, Dict[str, Any]]:
    return _get_articles_data(links, investors, default=default)


def marketwatch(link: str, default: Optional[Any] = None) -> Dict[str, Any]:
    res = requests.get(link)
    data = OrderedDict((field, default) for field in MARKETWATCH_FIELDS)
    try:
        res.raise_for_status()
    except requests.exceptions.HTTPError as err:
        logger.error(f"HTTP request failed for marketwatch article: {link}\nError:\n{err}")
        return data
    soup = BeautifulSoup(res.text, "html.parser")

    if breadcrumbs := soup.find("ol", attrs={"class": re.compile(".*breadcrumbs.*")}):
        try:
            data["categories"] = [
                breadcrumbs.find_all("li")[-1].text.strip(),
            ]
        except IndexError:
            logger.warning(f"Failed to find category for marketwatch article: {link}")
    else:
        logger.warning(f"Failed to find category breadcrumbs for marketwatch article: {link}")

    if article_meta := soup.find("div", attrs={"class": "article__masthead"}):
        if date := article_meta.find("time"):
            date_clean = date.text.replace(".", "").replace("Sept ", "Sep ").split(": ")[-1].strip()
            if date_obj := _validate_date(date_clean, "%B %d, %Y at %I:%M %p ET"):
                data["date"] = date_obj
            elif date_obj := _validate_date(date_clean, "%b %d, %Y at %I:%M %p ET"):
                data["date"] = date_obj
            else:
                logger.warning(f"Failed to parse date for marketwatch article: {link}, Date: {date_clean}")
        else:
            logger.warning(f"Failed to find date for marketwatch article: {link}")

        if author := article_meta.find("h4", attrs={"itemprop": "name"}):
            data["author"] = string.capwords(author.text)
        else:
            logger.warning(f"Failed to find author for marketwatch article: {link}")

        if title := article_meta.find("h1", attrs={"itemprop": "headline"}):
            data["title"] = string.capwords(title.text)
        else:
            logger.warning(f"Failed to find title for marketwatch article: {link}")
    else:
        logger.warning(f"Failed to find metadata for marketwatch article: {link}")

    if content := soup.find("div", attrs={"id": "js-article__body"}):
        if content_paragraphs := content.find_all("p"):
            data["content"] = "\n".join(du.get_paragraph(p) for p in content_paragraphs).strip()
        else:
            logger.warning(f"Failed to find content paragraph for marketwatch article: {link}")
    else:
        logger.warning(f"Failed to find content block for marketwatch article: {link}")

    if tickers := soup.find_all("span", attrs={"class": "symbol"}):
        data["tickers"] = {ticker.text.upper() for ticker in tickers}
    else:
        data["tickers"] = set()

    return data


def marketwatch_articles(links: List[str], default: Optional[Any] = None) -> Dict[str, Dict[str, Any]]:
    return _get_articles_data(links, marketwatch, default=default)


def investopedia(link: str, default: Optional[Any] = None) -> Dict[str, Any]:
    res = requests.get(link)
    data = OrderedDict((field, default) for field in INVESTOPEDIA_FIELDS)
    try:
        res.raise_for_status()
    except requests.exceptions.HTTPError as err:
        logger.error(f"HTTP request failed for investopedia article: {link}\nError:\n{err}")
        return data
    soup = BeautifulSoup(res.text, "html.parser")

    if breadcrumps := soup.find("div", attrs={"class": "comp breadcrumbs"}):
        if categories := breadcrumps.find_all("a", attrs={"class": "breadcrumbs__link"}):
            data["categories"] = [string.capwords(category.text) for category in categories]
        else:
            logger.warning(f"Failed to find categories for investopedia article: {link}")
    else:
        logger.warning(f"Failed to find breadcrumps for investopedia article: {link}")

    if article_meta := soup.find("header", attrs={"id": "article-header_1-0"}):
        if author := article_meta.find("a", attrs={"class": re.compile("byline__link.*")}):
            data["author"] = string.capwords(author.text)
        else:
            logger.warning(f"Failed to find author for investopedia article: {link}")

        if date := article_meta.find("div", attrs={"id": "displayed-date_1-0"}):
            if date_obj := _validate_date(date.text.strip(), "Updated %b %d, %Y"):
                data["date"] = date_obj
            else:
                logger.warning(f"Failed to parse date for investopedia article: {link}\nDate: {date.text.strip()}")
        else:
            logger.warning(f"Failed to find date for investopedia article: {link}")

        if title := article_meta.find("h1", attrs={"id": "article-heading_2-0"}):
            data["title"] = string.capwords(title.text)
        else:
            logger.warning(f"Failed to find title for investopedia article: {link}")

        if subtitle := article_meta.find("h2", attrs={"id": "article-subheading_1-0"}):
            data["subtitle"] = subtitle.text.strip()
        else:
            logger.warning(f"Failed to find subtitle for investopedia article: {link}")
    else:
        logger.warning(f"Failed to find metadata for investopedia article: {link}")

    if content := soup.find("div", attrs={"id": "article-body_1-0"}):
        if content_paragraphs := content.find_all("p"):
            data["content"] = "\n".join(du.get_paragraph(p) for p in content_paragraphs).strip()
        else:
            logger.warning(f"Failed to find content paragraph for investopedia article: {link}")
    else:
        logger.warning(f"Failed to find content block for investopedia article: {link}")

    if tickers := soup.find_all("a", attrs={"href": re.compile("https://www.investopedia.com/markets/quote.*")}):
        data["tickers"] = {ticker.text.strip().upper() for ticker in tickers}
    else:
        data["tickers"] = set()

    return data


def investopedia_articles(links: List[str], default: Optional[Any] = None) -> Dict[str, Dict[str, Any]]:
    return _get_articles_data(links, investopedia, default=default)


SOURCE_FUNCS = {
    "yahoo": yahoo,
    "fool": motley,
    "barrons": barrons,
    "investors": investors,
    "marketwartch": marketwatch,
    "investopedia": investopedia,
}


def multi_site_articles(links: List[str], default: Optional[Any] = None) -> Dict[str, Dict[str, Any]]:
    data = OrderedDict()
    for link in (pbar := tqdm(links, total=len(links))) :
        pbar.set_description(link)
        try:
            source = link.split("/")[2].split(".")[1]
        except IndexError:
            logger.warning(f"Failed to find source site for article: {link}")
            source = ""

        if source in SOURCE_FUNCS:
            data[link] = SOURCE_FUNCS[source](link, default=default)
        else:
            logger.warning(f"Failed to find a supported source site for article: {link}")
            data[link] = None
    return data
