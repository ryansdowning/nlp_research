import datetime
import re
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union

import bs4
import requests
from bs4 import BeautifulSoup
from loguru import logger

from nlp_utils import db_utils as dbu
from scrape_utils import data_utils as du

BASE = 'https://www.fool.com'
EARNINGS = '/earnings-call-transcripts'
NEWS = '/investing-news/?page=1'
TRANSCRIPT_FIELDS = ('publication_date', 'fiscal_quarter', 'fiscal_year',
                     'call_date', 'call_time', 'exchange', 'ticker', 'content')
ARTICLE_FIELDS = ('date', 'title', 'subtitle', 'content', 'tickers')


def get_earnings_calls() -> List[bs4.element.Tag]:
    """Gets all earning call hyperlinks from the current front page of motley fool earnings-call-transcripts

    Returns:
        List of 'a' href tags from beautiful soup
    """
    main_response = requests.get(f"{BASE}{EARNINGS}")
    main_soup = BeautifulSoup(main_response.text, 'html.parser')
    earnings_calls = main_soup.find_all('a', attrs={'data-id': 'article-list', 'href': True})
    return earnings_calls


def get_links(a_lst: List[bs4.element.Tag]) -> List[str]:
    """Gets links from a list of 'a' href objects from beautiful soup

    Args:
        a_lst: list of 'a' href beautiful soup tags

    Returns:
        list of strings corresponding to the respective hyperlinks contained within each 'a' tag
    """
    return [a['href'] for a in a_lst]


def next_element(elem: bs4.element.Tag) -> Union[bs4.element.Tag, None]:
    """Returns the next sibling of a beautiful soup tag

    Args:
        elem: Current beautiful soup tag element

    Returns:
         Next sibling of the given element - Returns None if no next sibling available
    """
    while elem is not None:
        # Find next element, skip NavigableString objects
        elem = elem.next_sibling
        if hasattr(elem, 'name'):
            return elem
    return None


def get_paragraph(header: bs4.element.Tag) -> str:
    """Extracts the text under a given header element. Continues until header of same level is found

    Args:
        header: beautiful soup tag of a header to start extracting text at

    Returns:
        string of all the text under the given header until the next header of equal level is found
    """
    page = [header.text]
    elem = next_element(header)
    while elem and elem.name != header.name:
        s = elem.string
        if s:
            page.append(s)
        elem = next_element(elem)
    return re.sub(r"\s{2,}", '\n', '\n'.join(page))


def get_transcript_data(transcript_link: str) -> Dict[str, Any]:
    """Uses beautiful soup facilities to extract standardized information from earnings call transcript

    Currently extracts: publication date, fiscal quarter, fiscal year, date and time of earnings call,
    company listed exchange, company ticker, company name, content of earnings call broken up by header

    Args:
        transcript_link: Sublink of earnings call transcript from motley fool

    Returns:
        dictionary containing information elements extracted from the given earnings call transcript
    """
    logger.info("Getting earnings transcript data for %s" % transcript_link)
    data = OrderedDict()
    call_response = requests.get(f"{BASE}{transcript_link}")
    call_soup = BeautifulSoup(call_response.text, 'html.parser')
    content_soup = call_soup.find(attrs={'class': 'article-content'})

    # Date and time of transcript publication
    pub_date = (call_soup.find(attrs={'class': 'publication-date'})).text.strip()
    data['publication_date'] = datetime.datetime.strptime(pub_date, '%b %d, %Y at %I:%M%p')

    # Earnings call information
    call_text = call_soup.text.replace('\xa0', ' ')
    try:
        quarter, year, call_date, call_time = re.findall(
            r"Q(\d) (\d{4}) Earnings Call([A-Z][a-z]{2}\.? \d{1,2}, \d{4}), (\d{1,2}:\d{2} (?:a\.m\.|p\.m\.) ET)",
            call_text
        )[0]
        data['fiscal_quarter'] = int(quarter)
        data['fiscal_year'] = int(year)
        data['call_date'] = datetime.datetime.strptime(call_date.replace('.', ''), '%b %d, %Y')
        data['call_time'] = call_time
    except IndexError:
        logger.warning("Failed to get earnings call information for: %s" % transcript_link)
        data['fiscal_quarter'] = None
        data['fiscal_year'] = None
        data['call_date'] = None
        data['call_time'] = None

    # Company information
    try:
        ticker_info = call_soup.find(attrs={'class': 'ticker', 'data-id': True}).text
        data['exchange'], data['ticker'] = re.findall(
            r"^\((NYSE|NYSEMKT|NASDAQ|OTC):([A-Z]+(?:\.[A-Z])?)\)$", ticker_info
        )[0]
    except (IndexError, AttributeError):
        logger.warning("Failed to get exchange and ticker from earnings call for %s:" % transcript_link)
        data['exchange'] = None
        data['ticker'] = None
    data['company'] = content_soup.strong.text

    # Format different content sections into dictionary
    data['content'] = OrderedDict()
    titles = sorted([li.text for li in content_soup.ul.find_all('li')])
    headers = content_soup.find_all('h2')
    if headers:
        headers = [header for header in content_soup.find_all('h2') if header.text[:-1] in titles]
        headers = sorted(headers, key=lambda header: header.text)
        for title, header in zip(titles, headers):
            data['content'][title] = get_paragraph(header)
    else:
        logger.warning("Failed to find content headers for: %s" % transcript_link)
    return data


def get_transcripts_data(transcript_links: List[str]) -> Dict[str, Dict[str, Any]]:
    """Extracts information for a list of transcripts using get_transcript_data

    Args:
        transcript_links: List of sublinks for earnings call transcripts from motley fool

    Returns:
        dictionary of data extracted from each transcript, keyed by the respective (sub)link
    """
    data = OrderedDict()
    for link in transcript_links:
        data[link] = get_article_data(link)
    return data


def get_recent_articles() -> List[str]:
    response = requests.get(f"{BASE}{NEWS}")
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    if stories := soup.find('div', attrs={'class': 'top-stories'}):
        story_data = dict()
        return [story['href'] for story in stories.find_all('a', attrs={'data-id': 'article-list'})]
    else:
        raise ValueError("Error encountered when fetching recent articles, article list not found!")


def get_articles_data(article_links: List[str]) -> Dict[str, Dict[str, Any]]:
    data = OrderedDict()
    for link in article_links:
        data[link] = get_transcript_data(link)
    return data


def get_article_data(article_link: str) -> Dict[str, Any]:
    article = f"{BASE}{article_link}"
    res = requests.get(article)
    if res.status_code != 200:
        if 200 < res.status_code < 300:
            logger.warning("Article fetch successful but has non 200 status code: %d ", res.status_code)
        else:
            logger.error("Article fetch failed with status code: %d ", res.status_code)
            res.raise_for_status()
    article_soup = BeautifulSoup(res.text, 'html.parser')
    data = OrderedDict()
    
    if pub_date := article_soup.find('div', attrs={'class': 'publication-date'}):
        data['date'] = datetime.datetime.strptime(pub_date.text.strip(), '%b %d, %Y at %I:%M%p')
    else:
        logger.warning("Failed to find date for article: %s", article_link)
        data['date'] = None
    
    if title := article_soup.find('div', attrs={'id': 'adv_text', 'class': 'adv-heading'}):
        try:
            title = title.next_sibling.next_sibling
            data['title'] = title.text.strip()
        except AttributeError as err:
            logger.warning("Failed to find to find title for article: %s", article_link)
            data['title'] = None
        try:
            data['subtitle'] = title.next_sibling.next_sibling.text.strip()
        except AttributeError as err:
            logger.warning("Failed to find subtitle for article: %s", article_link)
            data['subtitle'] = None
    else:
        logger.warning("Failed to find title and subtitle div for article: %s", article_link)
        data['subtitle'] = None
        data['title'] = None

    if author := article_soup.find('div', attrs={'class': 'author-name'}):
        if name := author.find('a'):
                data['author'] = name.text.strip()
        else:
            logger.warning("Failed to find author name for article: %s", article_link)
            data['author'] = None
    else:
        logger.warning("Failed to find author div for article: %s", article_link)
        data['author'] = None
    
    if article_content := article_soup.find('span', attrs={'class': 'article-content'}):
        article_text = re.sub(r"\n{2,}", r"\n", article_content.text.strip())
        data['content'] = article_text
    else:
        logger.warning("Failed to find content for article: %s", article_link)
        data['content'] = None
    
    tickers = article_content.find_all('span', attrs={'class': 'ticker'})
    tickers = [re.match(r"\((?P<exchange>[A-Z]+):(?P<ticker>[A-Z]+)\)", ticker.text.strip()) for ticker in tickers]
    tickers = {(ticker.group('exchange'), ticker.group('ticker')) for ticker in tickers if ticker}
    data['tickers'] = tickers

    return data

def transcript_stream(data_fields: Optional[List[str]] = None, update=60):
    """Creates generator object that will stream transcript data

    Args:
        data_fields: Ordered list of attributes to extract from transcripts. If none, will use all data fields
        update: Time in seconds to wait between pull requests

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
    visted_links = set()
    links = set(get_links(get_earnings_calls()))
    while True:
        while links:
            # Find next non-visited link
            next_link = links.pop()
            while links and next_link in visted_links:
                next_link = links.pop()
            # If not visted, yield the data of the transcript
            if next_link not in visted_links:
                visted_links.add(next_link)
                data = get_transcript_data(next_link)
                output = [next_link] + [data[field] for field in data_fields]
                yield tuple(output)
        # Sleep and get new earnings call transcript links
        time.sleep(update)
        links = set(get_links(get_earnings_calls())) - visted_links


def stream_earnings_transcripts(
        data_fields: Optional[List[str]] = None,
        update=60,
        file: Optional[str] = None,
        table: Optional[dbu.DBTable] = None,
        **kwargs
):
    """Creates a process that indefinitely streams data from earnings call transcripts to a csv

    Args:
        file: Name of file (csv) to write data to
        table: DBTable object that can be provided to insert data directly into SQL database
        data_fields: Ordered list of attributes to extract from transcripts. If none, will use all data fields
        update: Time in seconds to wait between pull requests

    Returns:
        None - Process runs until error is thrown or is interrupted
        Data stream is written to the given file
    """
    stream = transcript_stream(data_fields, update)
    du.stream_data(stream, data_fields, file, table, **kwargs)
