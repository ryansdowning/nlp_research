from bs4 import BeautifulSoup
import requests
from collections import defaultdict
import re
import datetime

BASE = 'https://www.fool.com'
MAIN = '/earnings-call-transcripts'


def get_earnings_calls():
    main_response = requests.get(f"{BASE}{MAIN}")
    main_soup = BeautifulSoup(main_response.text, 'html.parser')
    earnings_calls = main_soup.find_all('a', attrs={'data-id': 'article-list', 'href': True})
    return earnings_calls


def get_links(a_lst):
    return [a['href'] for a in a_lst]


def next_element(elem):
    while elem is not None:
        # Find next element, skip NavigableString objects
        elem = elem.next_sibling
        if hasattr(elem, 'name'):
            return elem
    return None


def get_paragraph(header):
    page = [header.text]
    elem = next_element(header)
    while elem and elem.name != header.name:
        s = elem.string
        if s: page.append(s)
        elem = next_element(elem)
    return re.sub("\s{2,}", '\n', '\n'.join(page))


def get_transcript_data(transcript_links):
    data = dict()
    for link in transcript_links:
        data[link] = dict()
        call_response = requests.get(f"{BASE}{link}")
        call_soup = BeautifulSoup(call_response.text, 'html.parser')
        content_soup = call_soup.find(attrs={'class': 'article-content'})

        pub_date = (call_soup.find(attrs={'class': 'publication-date'})).text.strip()
        data[link]['publication_date'] = datetime.datetime.strptime(pub_date, '%b %d, %Y at %I:%M%p')
        quarter, year, call_date, call_time = re.findall(
            "Q(\d{1}).*(\d{4}) Earnings Call([A-Z][a-z]{2} \d{1,2}, \d{4}), (\d{1,2}:\d{2} (?:a\.m\.|p\.m\.) ET)",
            call_soup.text
        )[0]
        data[link]['fiscal_quarter'] = int(quarter)
        data[link]['fiscal_year'] = int(year)
        data[link]['call_date'] = datetime.datetime.strptime(call_date, '%b %d, %Y')
        data[link]['call_time'] = call_time

        ticker_info = call_soup.find(attrs={'class': 'ticker', 'data-id': True}).text
        data[link]['exchange'], data[link]['ticker'] = re.findall("\((NYSE|NYSEMKT|NASDAQ):([A-Z]+)\)", ticker_info)[0]
        data[link]['company'] = content_soup.strong.text

        data[link]['content'] = dict()
        titles = [li.text for li in content_soup.ul.find_all('li')]
        headers = content_soup.find_all('h2')[1:]
        if len(titles) != len(headers):
            raise ValueError("Did not find header for every section in content")
        for title, header in zip(titles, headers):
            data[link]['content'][title] = get_paragraph(header)
    return data
