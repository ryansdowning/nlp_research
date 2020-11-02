import datetime
import re
from typing import Any, Dict, List, Union

import bs4
import requests
from bs4 import BeautifulSoup

BASE = 'https://www.fool.com'
MAIN = '/earnings-call-transcripts'


def get_earnings_calls() -> List[bs4.element.Tag]:
    """Gets all earning call hyperlinks from the current front page of motley fool earnings-call-transcrips

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
    data = dict()
    call_response = requests.get(f"{BASE}{transcript_link}")
    call_soup = BeautifulSoup(call_response.text, 'html.parser')
    content_soup = call_soup.find(attrs={'class': 'article-content'})

    pub_date = (call_soup.find(attrs={'class': 'publication-date'})).text.strip()
    data['publication_date'] = datetime.datetime.strptime(pub_date, '%b %d, %Y at %I:%M%p')
    quarter, year, call_date, call_time = re.findall(
        r"Q(\d).*(\d{4}) Earnings Call([A-Z][a-z]{2} \d{1,2}, \d{4}), (\d{1,2}:\d{2} (?:a\.m\.|p\.m\.) ET)",
        call_soup.text
    )[0]
    data['fiscal_quarter'] = int(quarter)
    data['fiscal_year'] = int(year)
    data['call_date'] = datetime.datetime.strptime(call_date, '%b %d, %Y')
    data['call_time'] = call_time

    ticker_info = call_soup.find(attrs={'class': 'ticker', 'data-id': True}).text
    data['exchange'], data['ticker'] = re.findall(r"\((NYSE|NYSEMKT|NASDAQ):([A-Z]+)\)", ticker_info)[0]
    data['company'] = content_soup.strong.text

    data['content'] = dict()
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
    return {link: get_transcript_data(link) for link in transcript_links}
