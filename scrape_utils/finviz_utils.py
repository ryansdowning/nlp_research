import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import datetime
import time
from typing import Optional, List, Dict, Any, Union, Callable, Iterable, Tuple
from nlp_utils import db_utils as dbu
from scrape_utils import data_utils as du

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko)'
                  ' Chrome/58.0.3029.110 Safari/537.36',
    'Upgrade-Insecure-Requests': '1',
    'Cookie': 'v2=1495343816.182.19.234.142',
    'Accept-Encoding': 'gzip, deflate, sdch',
    'Referer': "http://finviz.com/quote.ashx?t="
}
BASE_URL = "https://finviz.com/quote.ashx?t="
TICKER_COL = 'ticker'

NEWS_COLS = ('date', 'title', 'link') + (TICKER_COL,)
NEWS_PRIMARY_COLS = ('link', TICKER_COL)
NEWS_PRIMARY_IDX = tuple(NEWS_COLS.index(col) for col in NEWS_PRIMARY_COLS)

RATINGS_COLS = ('date', 'update', 'analyst', 'position', 'target', 'low_target', 'high_target') + (TICKER_COL,)
RATINGS_PRIMARY_COLS = ('date', 'analyst', TICKER_COL)
RATINGS_PRIMARY_IDX = tuple(RATINGS_COLS.index(col) for col in RATINGS_PRIMARY_COLS)


def get_ticker_response(ticker: str) -> requests.models.Response:
    """Processes get request for given ticker on Finviz

    Args:
        ticker: ticker to get finviz webpage response for

    Returns:
        response object
    """
    response = requests.get(f"{BASE_URL}{ticker}", headers=headers)
    response.raise_for_status()
    return response


def get_ticker_soup(ticker: str) -> BeautifulSoup:
    """Creates BeautifulSoup object from get request for a ticker on Finviz

    Args:
        ticker: ticker to get BeautifulSoup for

    Returns:
        BeautifulSoup object
    """
    response = get_ticker_response(ticker)
    return BeautifulSoup(response.text, 'html.parser')


def get_ticker_info(ticker: str) -> Dict[str, Any]:
    """Conglomerate of all scraping functionality for finviz tickers

    Args:
        ticker: ticker to get finviz data for

    Returns:
        dictionary with information fields and their corresponding data
    """
    soup = get_ticker_soup(ticker)
    ratings_df = get_ratings_df(soup)
    news_df = get_news_df(soup)
    return {'ratings': ratings_df, 'news': news_df}


def get_news_df(ticker_soup: BeautifulSoup) -> pd.DataFrame:
    """Scrape news table from finviz page and format into pandas dataframe

    Args:
        ticker_soup: BeautifulSoup object from the finviz page

    Returns:
        pandas dataframe that contains information from the articles found in the news table of the soup object
    """
    news_table = ticker_soup.find("table", {"class": "fullview-news-outer"})
    table = []
    rows = news_table.find_all("tr")

    recent_date = None  # Finviz only lists date when it changes so we have to keep track of the last seen date
    for tr in rows:
        td = tr.find_all('td')  # Get elements from row of news table
        date_info = td[0].text.replace('\xa0', "").split(' ')

        # If date_info has 2 elements, the first is the new "recent_date"
        if len(date_info) == 2:
            recent_date = datetime.datetime.strptime(date_info.pop(0), "%b-%d-%y").date()

        # The only element in date_info at this point is the time, since date was not present, or was popped out above
        news_time = datetime.datetime.strptime(date_info[0], "%I:%M%p").time()
        combined_date = datetime.datetime.combine(recent_date, news_time)

        news_source = tr.find_all('a')[0]  # Get news link element
        news_title = news_source.text
        news_link = news_source['href']
        table.append([combined_date, news_title, news_link])

    return pd.DataFrame(table, columns=NEWS_COLS[:-1])


def get_ratings_df(ticker_soup: BeautifulSoup) -> pd.DataFrame:
    """Scrape analyst ratings table from finviz page and format into pandas dataframe

    Args:
        ticker_soup: BeautifulSoup object from the finviz page

    Returns:
        pandas dataframe that contains information from the analyst ratings found in the soup object
    """
    ratings_table = ticker_soup.find("table", {"class": "fullview-ratings-outer"})
    table = []
    rows = ratings_table.find_all("tr")
    for tr in rows:
        td = tr.find_all('td')   # Get elements from row of ratings table
        if not td or td[0].has_attr('class'):  # Some rows are empty or duplicates with 'class' attribute
            continue
        row = [i.text for i in td]  # Extract text from each element
        table.append(row)
    df = pd.DataFrame(table, columns=['date', 'update', 'analyst', 'position', 'target'])
    df['date'] = pd.to_datetime(df['date'])

    def _get_low_high(targets: List[float]) -> Tuple[float, float]:
        """Splits analyst target range to low and high targets

        Args:
            targets: List of price targets

        Returns:
            tuple of low and high price targets
        """
        if not targets or pd.isna(targets).any():  # If empty or NaN target, return NaN low and high
            return np.nan, np.nan
        if len(targets) == 1:  # If only 1 target, set low and high to that target
            return targets[0], targets[0]
        if len(targets) == 2:  # If 2 targets, this is the low and high, so just sort them
            return tuple(sorted(targets))
        return np.nan, np.nan  # If >2 targets, undefined behavior so return NaN for high and low

    # Convert target to low and high price targets
    temp = df['target'].str.extractall(r"\$([0-9]*\.?[0-9]+)").groupby(level=0)[0].apply(
        lambda x: list(map(float, list(x)))
    ).apply(_get_low_high)
    targets_df = pd.DataFrame(temp.tolist(), columns=['low_target', 'high_target'], index=temp.index)
    df = df.join(targets_df, how='left')
    # TODO: Overwriting column names would cause problems if changes are made which defeats the purpose of constants
    df.columns = RATINGS_COLS[:-1]
    return df


def _multiple_tickers_df(
        tickers: List[str], df_func: Callable[[BeautifulSoup], pd.DataFrame], **kwargs
) -> pd.DataFrame:
    """Helper function to process a list of tickers for a given function. Said function must accept a BeautifulSoup
    object as its first parameter, with any other parameters specified in kwargs. The BeautifulSoup object is a response
    from the finviz webpage for each respective ticker passed in. A new <TICKER_COL> column is added to the final output
    which combines all of the results into a single dataframe where <TICKER_COL> shows the corresponding ticker.

    Args:
        tickers: List of tickers to process
        df_func: Callable that returns pandas dataframe. Must accept BeautifulSoup object as its first
                 (and only required) parameter
        kwargs: Any additional keyword arguments to be passed to df_func on each call

    Returns:
        pandas dataframe of the combined results with an additional <TICKER_COL> column
    """
    soups = [get_ticker_soup(ticker) for ticker in tickers]
    func_dfs = []
    for ticker, soup in zip(tickers, soups):
        func_df = df_func(soup, **kwargs)
        func_df[TICKER_COL] = ticker
        func_dfs.append(func_df)
    return pd.concat(func_dfs, axis=0)


def multiple_tickers_news_df(tickers: List[str]) -> pd.DataFrame:
    return _multiple_tickers_df(tickers, get_news_df)


def multiple_tickers_ratings_df(tickers: List[str]) -> pd.DataFrame:
    return _multiple_tickers_df(tickers, get_ratings_df)


def _get_indexes(data: List[Any], indexes: Iterable[int]) -> tuple:
    return tuple(data[idx] for idx in indexes)


def _ticker_func_stream(
        ticker: Union[List[str], str],
        df_func: Callable[[BeautifulSoup], pd.DataFrame],
        primary_indexes: Iterable[int],
        field_indexes: Iterable[int],
        update: int = 60
):
    if isinstance(ticker, str):
        ticker = [ticker]
    elif not isinstance(ticker, list):
        raise AttributeError(
            f"ticker argument must be a ticker string or list of ticker strings. Got unexpected type: {type(ticker)}"
        )

    visited = set()
    data = _multiple_tickers_df(ticker, df_func).values.tolist()
    while True:
        while data:
            next_data = data.pop()
            unique = _get_indexes(next_data, primary_indexes)
            while data and unique in visited:
                next_data = data.pop()
                unique = _get_indexes(next_data, primary_indexes)

            if unique not in visited:
                visited.add(unique)
                output = _get_indexes(next_data, field_indexes)
                yield output
        time.sleep(update)
        data = _multiple_tickers_df(ticker, df_func).values.tolist()


def ticker_news_stream(ticker: Union[List[str], str], data_fields: Optional[List[str]] = None, update: int = 60):
    if data_fields is None:
        data_fields = list(NEWS_COLS)
    elif not set(data_fields).issubset(set(NEWS_COLS)):
        raise AttributeError(
            f"Unexpected data fields for subreddit submissions:"
            f" {set(data_fields) - set(NEWS_COLS)}"
        )
    field_idxs = [NEWS_COLS.index(field) for field in data_fields]
    return _ticker_func_stream(ticker, get_news_df, NEWS_PRIMARY_IDX, field_idxs, update)


def ticker_ratings_stream(ticker: Union[List[str], str], data_fields: Optional[List[str]] = None, update: int = 60):
    if data_fields is None:
        data_fields = list(RATINGS_COLS)
    elif not set(data_fields).issubset(set(RATINGS_COLS)):
        raise AttributeError(
            f"Unexpected data fields for subreddit submissions:"
            f" {set(data_fields) - set(RATINGS_COLS)}"
        )
    field_idxs = [RATINGS_COLS.index(field) for field in data_fields]
    return _ticker_func_stream(ticker, get_ratings_df, RATINGS_PRIMARY_IDX, field_idxs, update)


def stream_ticker_news(
        ticker: Union[List[str], str],
        data_fields: Optional[List[str]] = None,
        update: int = 60,
        file: Optional[str] = None,
        table: Optional[dbu.DBTable] = None,
        **kwargs
):
    stream = ticker_news_stream(ticker, data_fields, update)
    du.stream_data(stream, data_fields, file, table, **kwargs)


def stream_ticker_ratings(
        ticker: Union[List[str], str],
        data_fields: Optional[List[str]] = None,
        update: int = 60,
        file: Optional[str] = None,
        table: Optional[dbu.DBTable] = None,
        **kwargs
):
    stream = ticker_ratings_stream(ticker, data_fields, update)
    du.stream_data(stream, data_fields, file, table, **kwargs)
