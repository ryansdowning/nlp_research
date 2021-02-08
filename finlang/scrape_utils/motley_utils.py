import datetime
import re
import string
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import bs4
import requests
from bs4 import BeautifulSoup
from loguru import logger
from tqdm import tqdm

from finlang.nlp_utils import db_utils as dbu
from finlang.scrape_utils import data_utils as du

BASE = 'https://www.fool.com'
EARNINGS = '/earnings-call-transcripts'
NEWS = '/investing-news/?page=1'
TRANSCRIPT_FIELDS = ('publication_date', 'fiscal_quarter', 'fiscal_year',
                     'call_date', 'exchange', 'ticker', 'content')
ARTICLE_FIELDS = ('category', 'author', 'date', 'title', 'subtitle', 'content', 'tickers')
EXCH_TICKER_RE = r"\((?P<exchange>[a-zA-Z]+):\s*(?P<ticker>[A-Z\.\-]+)\)"


def get_earnings_calls() -> List[str]:
    """Gets all earning call hyperlinks from the current front page of motley fool earnings-call-transcripts

    Returns:
        List of sublinks for earnings calls from motley fool
    """
    main_response = requests.get(f"{BASE}{EARNINGS}")
    try:
        main_response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        logger.error(f"HTTP requests failed for earnings\nError:\n{err}")
        return []
    main_soup = BeautifulSoup(main_response.text, 'html.parser')
    if earnings_calls := main_soup.find_all('a', attrs={'data-id': 'article-list', 'href': True}):
        return [call['href'] for call in earnings_calls]
    logger.error("Error encountered when fetching recent earnings calls, links not found!")
    return []


def get_links(a_lst: List[bs4.element.Tag]) -> List[str]:
    """Gets links from a list of 'a' href objects from beautiful soup

    Args:
        a_lst: list of 'a' href beautiful soup tags

    Returns:
        list of strings corresponding to the respective hyperlinks contained within each 'a' tag
    """
    return [a['href'] for a in a_lst]


def get_transcript_data(transcript_link: str, default: Optional[Any] = None) -> Dict[str, Any]:
    """Uses beautiful soup facilities to extract standardized information from earnings call transcript

    Currently extracts: publication date, fiscal quarter, fiscal year, date and time of earnings call,
    company listed exchange, company ticker, company name, content of earnings call broken up by header

    Args:
        transcript_link: Sublink of earnings call transcript from motley fool
        default: If failed to extract a data field, what value to store in its place. Default, None

    Returns:
        dictionary containing information elements extracted from the given earnings call transcript
    """
    logger.info(f"Getting earnings transcript data for {transcript_link}")
    data = OrderedDict((field, default) for field in TRANSCRIPT_FIELDS)
    call_response = requests.get(f"{BASE}{transcript_link}")
    try:
        call_response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        logger.error(f"HTTP request failed for transcript: {transcript_link}\nError:\n{err}")
        return data
    call_soup = BeautifulSoup(call_response.text, 'html.parser')
    content_soup = call_soup.find(attrs={'class': 'article-content'})

    # Date and time of transcript publication
    if pub_date := call_soup.find('div', attrs={'class': 'publication-date'}):
        data['publication_date'] = datetime.datetime.strptime(pub_date.text.strip(), '%b %d, %Y at %I:%M%p')
    else:
        logger.warning(f"Failed to find date for article: {transcript_link}")

    # Earnings call information
    call_text = call_soup.text.replace('\xa0', ' ')
    if call_info := re.search(
            r"Q(?P<quarter>\d) (?P<year>\d{4}) Earnings Call(?P<date>[A-Z][a-z]{2}\.? \d{1,2}, \d{4}), "
            r"(?P<time>\d{1,2}:\d{2} (?:a\.m\.|p\.m\.) ET)",
            call_text
    ):
        data['fiscal_quarter'] = int(call_info.group('quarter'))
        data['fiscal_year'] = int(call_info.group('year'))
        call_datetime = f"{call_info.group('date')} {call_info.group('time')}".replace('.', '')
        data['call_date'] = datetime.datetime.strptime(call_datetime.replace('.', ''), '%b %d, %Y %H:%M %p ET')
    else:
        logger.warning(f"Failed to get earnings call information for: {transcript_link}")

    # Company information
    ticker_info = call_soup.find(attrs={'class': 'ticker', 'data-id': True}).text
    if ticker_info := re.match(EXCH_TICKER_RE, ticker_info.strip()):
        data['ticker'] = ticker_info.group('ticker')
        data['exchange'] = ticker_info.group('exchange')
    else:
        logger.warning(f"Failed to get exchange and ticker from earnings call for: {transcript_link}")
    data['company'] = content_soup.strong.text

    # Format different content sections into dictionary
    data['content'] = OrderedDict()
    headers = content_soup.find_all('h2')
    if headers:
        for header in headers:
            key = header.text.casefold().replace(':', '').replace('&', 'and')
            data['content'][key] = du.get_paragraph(header)
    else:
        logger.warning(f"Failed to find content headers for: {transcript_link}")
    return data


def get_transcripts_data(transcript_links: List[str]) -> Dict[str, Dict[str, Any]]:
    """Extracts information for a list of transcripts using get_transcript_data

    Args:
        transcript_links: List of sublinks for earnings call transcripts from motley fool

    Returns:
        dictionary of data extracted from each transcript, keyed by the respective (sub)link
    """
    data = OrderedDict()
    for link in pbar := tqdm(transcript_links, total=len(transcript_links)):
        pbar.set_description(link)
        data[link] = get_transcript_data(link)
    pbar.close()
    return data


def get_recent_articles() -> List[str]:
    """Gets the recent articles from the front page of motley fools

    Returns:
        List[str]: list of article sublinks. Returns empty list if failed to fetch articles.
    """
    response = requests.get(f"{BASE}{NEWS}")
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        logger.error(f"HTTP request failed for recent articles\nError:\n{err}")
        return []
    soup = BeautifulSoup(response.text, 'html.parser')

    if stories := soup.find('div', attrs={'class': 'top-stories'}):
        return [story['href'] for story in stories.find_all('a', attrs={'data-id': 'article-list'})]
    logger.error("Error encountered when fetching recent articles, article list not found!")
    return []


def get_articles_data(article_links: List[str]) -> Dict[str, Dict[str, Any]]:
    """Extracts information from the provided motley fool articles

    Args:
        article_links (List[str]): List of sublinks to articles

    Returns:
        Dict[str, Dict[str, Any]]: Nested dictionary keyed on article link where the inner dictionary contains 
        information from that article
    """
    data = OrderedDict()
    for link in pbar := tqdm(article_links, total=len(article_links)):
        pbar.set_description(link)
        data[link] = get_article_data(link)
    pbar.close()
    return data


def get_article_data(article_link: str, default: Optional[Any] = None) -> Dict[str, Any]:
    """Extracts information from the given motley fools article

    Args:
        article_link (str): Sublink directing to specific article
        default: If failed to extract a data field, what value to store in its place. Default, None

    Returns:
        Dict[str, Any]: dictionary of data elements extracted from the article
    """
    article = f"{BASE}{article_link}"
    res = requests.get(article)
    data = OrderedDict((field, default) for field in ARTICLE_FIELDS)
    try:
        res.raise_for_status()
    except requests.exceptions.HTTPError as err:
        logger.error(f"HTTP request failed for article: {article_link}\nError:\n{err}")
        return data
    article_soup = BeautifulSoup(res.text, 'html.parser')
    data['category'] = article_link.split('/')[1]

    # Millionacres articles have a slightly different format
    if data['category'] == "millionacres":
        return _get_millionacres_data(article_soup, article_link, data)

    if pub_date := article_soup.find('div', attrs={'class': 'publication-date'}):
        date_clean = pub_date.text.replace('Updated:', '').strip()
        try:
            data['date'] = datetime.datetime.strptime(date_clean, '%b %d, %Y at %I:%M%p')
        except ValueError:
            logger.warning(f"Failed to parse date: [{date_clean}] from article: {article_link}")
    else:
        logger.warning(f"Failed to find date for article: {article_link}")

    if title := article_soup.find('div', attrs={'id': 'adv_text', 'class': 'adv-heading'}):
        try:
            title = title.next_sibling.next_sibling
            data['title'] = string.capwords(title.text)
        except AttributeError:
            logger.warning(f"Failed to find to find title for article: {article_link}")
        try:
            data['subtitle'] = string.capwords(title.next_sibling.next_sibling.text)
        except AttributeError:
            logger.warning(f"Failed to find subtitle for article: {article_link}")
    else:
        logger.warning(f"Failed to find title and subtitle div for article: {article_link}")

    if author := article_soup.find('div', attrs={'class': 'author-name'}):
        if name := author.find('a'):
            data['author'] = string.capwords(name.text)
        else:
            logger.warning(f"Failed to find author name for article: {article_link}")
    else:
        logger.warning(f"Failed to find author div for article: {article_link}")

    if article_content := article_soup.find('span', attrs={'class': 'article-content'}):
        article_text = re.sub(r"\n{2,}", r"\n", article_content.text.strip())
        data['content'] = article_text
        tickers = article_content.find_all('span', attrs={'class': 'ticker'})
        tickers = [re.match(EXCH_TICKER_RE, ticker.text.strip()) for ticker in tickers]
        tickers = {(ticker.group('exchange'), ticker.group('ticker')) for ticker in tickers if ticker}
        data['tickers'] = tickers
    else:
        logger.warning(f"Failed to find content for article: {article_link}")

    return data


def _get_millionacres_data(article_soup: BeautifulSoup, link: str, data: OrderedDict) -> Dict[str, Any]:
    """Helper function to extract information from motley fools articles in the millionacres category which is formatted
    differently from the other articles

    Args:
        article_soup: beautifulsoup object for the given article
        link: link of the corresponding article
        data: Data dictionary to store extracted information

    Returns:
        Dict[str, Any]: dictionary of data elements extracted from the article
    """
    if date_author := article_soup.find('div', attrs={'class': re.compile(r"author-and-date[a-z0-9\s\-]*")}):
        try:
            date, author = date_author.text.split("by")
            try:
                data['date'] = datetime.datetime.strptime(date.strip(), '%b %d, %Y')
            except ValueError:
                logger.warning(f"Failed to parse date for millionacres article: {link}")
            data['author'] = string.capwords(author)
        except ValueError:
            logger.warning(f"Failed to parse date and author for millionacres article: {link}")

    if title := article_soup.find('h1', attrs={'class': re.compile(r"article-header[a-z0-9\s\-]*")}):
        data['title'] = string.capwords(title.text)

    if article := article_soup.find('article', attrs={'class': re.compile("main.*")}):
        if content := article.find('div', attrs={'class': 'block-paragraph'}):
            if content_paragraphs := content.find_all('p'):
                data['content'] = "\n".join(du.get_paragraph(p) for p in content_paragraphs)
            else:
                logger.warning(f"Failed to find content paragraphs for millionacres article: {link}")
        else:
            logger.warning(f"Failed to find content block for millionacres article: {link}")
    else:
        logger.warning(f"Failed to find article content for millionacres article: {link}")

    if data['content']:
        data['tickers'] = set(re.findall(EXCH_TICKER_RE, data['content']))
    else:
        data['tickers'] = set()

    return data


def transcript_stream(data_fields: Optional[List[str]] = None, update: int = 60):
    """Creates generator object that will stream transcript data

    Args:
        data_fields (List[str], optional): Ordered list of attributes to extract from transcripts.
                                           If none, will use all data fields
        update (int): Time in seconds to wait between pull requests. Defaults to 60 seconds.

    Returns:
        Generator object that will yield tuples of length equal to data fields
    """
    if data_fields is None:
        data_fields = list(TRANSCRIPT_FIELDS)
    elif not set(data_fields).issubset(TRANSCRIPT_FIELDS):
        raise AttributeError(
            f"Unexpected data fields for subreddit submissions:"
            f" {set(data_fields) - set(TRANSCRIPT_FIELDS)}"
        )
    visited_links = set()
    links = set(get_earnings_calls())
    while True:
        while links:
            # Find next non-visited link
            next_link = links.pop()
            while links and next_link in visited_links:
                next_link = links.pop()
            # If not visited, yield the data of the transcript
            if next_link not in visited_links:
                visited_links.add(next_link)
                data = get_transcript_data(next_link)
                output = [next_link] + [data[field] for field in data_fields]
                yield tuple(output)
        # Sleep and get new earnings call transcript links
        time.sleep(update)
        links = set(get_earnings_calls()) - visited_links


def stream_earnings_transcripts(
        data_fields: Optional[List[str]] = None,
        update: int = 60,
        file: Optional[str] = None,
        table: Optional[dbu.DBTable] = None,
        **kwargs
):
    """Creates a process that indefinitely streams data from earnings call transcripts to a csv

    Args:
        data_fields (List[str], optional): Ordered list of attributes to extract from transcripts.
                                           If none, will use all data fields. Defaults to None.
        update (int, optional): Time in seconds to wait between pull requests. Defaults to 60.
        file (str, optional): Name of file (csv) to write data to. Defaults to None.
        table (dbu.DBTable, optional): DBTable object that can be provided to insert data directly into SQL database.
                                       Defaults to None.
        kwargs: Additional keyword arguments for streaming data
        
    Returns:
        None - Process runs until error is thrown or is interrupted
        Data stream is written to the given file
    """
    stream = transcript_stream(data_fields, update)
    du.stream_data(stream, data_fields, file, table, **kwargs)
