import datetime
import re
import time
from typing import Any, Dict, List, Union, Optional
from collections import OrderedDict

import bs4
import csv
import requests
from loguru import logger
from bs4 import BeautifulSoup

from scrape_utils import data_utils as du

BASE = 'https://www.fool.com'
MAIN = '/earnings-call-transcripts'
TRANSCRIPT_FIELDS = ('publication_date', 'fiscal_quarter', 'fiscal_year',
                     'call_date', 'call_time', 'exchange', 'ticker', 'content')


def get_earnings_calls() -> List[bs4.element.Tag]:
    """Gets all earning call hyperlinks from the current front page of motley fool earnings-call-transcripts

    Returns:
        List of 'a' href tags from beautiful soup
    """
    main_response = requests.get(f"{BASE}{MAIN}")
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
    data = OrderedDict()
    call_response = requests.get(f"{BASE}{transcript_link}")
    call_soup = BeautifulSoup(call_response.text, 'html.parser')
    content_soup = call_soup.find(attrs={'class': 'article-content'})

    pub_date = (call_soup.find(attrs={'class': 'publication-date'})).text.strip()
    data['publication_date'] = datetime.datetime.strptime(pub_date, '%b %d, %Y at %I:%M%p')
    try:
        quarter, year, call_date, call_time = re.findall(
            r"Q(\d) (\d{4}) Earnings Call([A-Z][a-z]{2}\.? \d{1,2}, \d{4}), (\d{1,2}:\d{2} (?:a\.m\.|p\.m\.) ET)",
            call_soup.text.replace('\xa0', ' ')
        )[0]
        data['fiscal_quarter'] = int(quarter)
        data['fiscal_year'] = int(year)
        data['call_date'] = datetime.datetime.strptime(call_date.replace('.', ''), '%b %d, %Y')
        data['call_time'] = call_time
    except IndexError:
        data['fiscal_quarter'] = None
        data['fiscal_year'] = None
        data['call_date'] = None
        data['call_time'] = None

    ticker_info = call_soup.find(attrs={'class': 'ticker', 'data-id': True}).text
    try:
        data['exchange'], data['ticker'] = re.findall(r"\((NYSE|NYSEMKT|NASDAQ|OTC):([A-Z]+\.?[A-Z])\)", ticker_info)[0]
    except IndexError:
        data['exchange'] = None
        data['ticker'] = None
    data['company'] = content_soup.strong.text

    data['content'] = OrderedDict()
    titles = [li.text for li in content_soup.ul.find_all('li')]
    headers = content_soup.find_all('h2')[1:]
    if len(titles) != len(headers):
        raise ValueError("Did not find header for every section in content")
    for title, header in zip(titles, headers):
        data['content'][title] = get_paragraph(header)
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
        data[link] = get_transcript_data(link)
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


def stream_earnings_transcripts(file, data_fields: Optional[List[str]] = None, update=60):
    """Creates a process that indefinitely streams data from earnings call transcripts to a csv

    Args:
        file: Name of file (csv) to write data to
        data_fields: Ordered list of attributes to extract from transcripts. If none, will use all data fields
        update: Time in seconds to wait between pull requests

    Returns:
        None - Process runs until error is thrown or is interrupted
        Data stream is written to the given file
    """
    if data_fields is None:
        data_fields = list(TRANSCRIPT_FIELDS)
    elif not set(data_fields).issubset(TRANSCRIPT_FIELDS):
        raise AttributeError(
            f"Unexpected data fields for subreddit submissions:"
            f" {set(data_fields) - set(TRANSCRIPT_FIELDS)}"
        )
    du.open_file_stream(file, data_fields)
    for data in transcript_stream(data_fields, update):
        with open(file, 'a', encoding='utf-8') as out:
            logger.info('writing %s data to to file at: %s' % (data[0], file))
            csv_out = csv.writer(out)
            csv_out.writerow(data)