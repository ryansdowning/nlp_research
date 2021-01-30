import argparse
import time
from functools import partial

from loguru import logger

from sentiment.config.db_config import conn
from sentiment.nlp_utils import db_utils as dbu
from sentiment.scrape_utils import finviz_utils as fu
from sentiment.scrape_utils import motley_utils as mu
from sentiment.scrape_utils import reddit_utils as ru

# Keeps the stream from spamming an API target or bad code from repeatedly erroring out
# This will not ever terminate the worker, the user should monitor their logs to ensure it is running stably
CD_TIMEOUT = 60
CD_LIMIT = 10
CD_ATTEMPTS = 5
CD_TIMER = [time.time()] * CD_ATTEMPTS
STREAMS = ("reddit_comments", "reddit_submissions", "finviz_news", "finviz_ratings", "motley_earnings", "motley_news")

parser = argparse.ArgumentParser()

# BASE STREAM ARGUMENTS
parser.add_argument(
    "-s", "--stream", type=str, help=f"Stream name for scraping data, currently supports: {STREAMS}", choices=STREAMS
)
parser.add_argument(
    "--fields",
    default=None,
    nargs="*",
    type=str,
    help="Which fields from the stream to include in the output stream. Default None, all fields are included. "
    "Must be a subset of the given streams fields.",
)
parser.add_argument("-f", "--file", default=None, type=str, help="Path of comments output file, must be csv")
parser.add_argument(
    "-t",
    "--table",
    default=None,
    type=str,
    help="Name of table in database to write comments data to. Will use the connection parameters specified in"
    " db_config to access the database",
)
parser.add_argument(
    "-k",
    "--keywords",
    default=None,
    nargs="*",
    type=str,
    help="List of strings, at least one of which must be found in the <keyword_fields> of each row in order for the"
    " entry to be streamed to the provided output(s). Default None, keyword filtering is ignored. If provided,"
    " must also supply --keyword_fields and optionally --case_sensitive",
)
parser.add_argument(
    "--keyword_fields",
    default=None,
    nargs="*",
    type=str,
    help="If filtering keywords, list of field names to check for keywords. "
    "Default None, must be provided if --keywords is specified",
)
parser.add_argument(
    "--case_sensitive",
    default=True,
    type=bool,
    help="If filtering keywords, whether or not the provided keywords should be treated case sensitively, "
    "provided as a boolean, default True",
)
parser.add_argument(
    "--timestamp",
    action="store_true",
    help="If provided, the stream will include an entry for the timestamp of the streamed data in each row",
)
parser.add_argument(
    "-v",
    "--verbose",
    action="count",
    default=2,
    help="Level of verbosity provided as an integer 0-4: OFF (O) DEBUG (1) INFO (2) WARNING (3) ERROR (4), default 2. "
    "Will log to file in current director with timestamped start time name",
)

# Reddit argument(s)
parser.add_argument(
    "--subreddit",
    default=None,
    type=str,
    help="Which subreddit to stream data for when streaming reddit comments or submissions, provided as a string",
)

# Finviz / Motley
parser.add_argument(
    "-u",
    "--update",
    default=60,
    type=int,
    help="How often to rescan the webpages for new data when streaming from Finviz or Motley,"
    " provided as an integer (seconds), default 60",
)
# Finviz
parser.add_argument(
    "--tickers",
    default=None,
    nargs="*",
    type=str,
    help="Tickers to scan for when streaming finviz news or ratings, provided as 1+ string symbols (all caps)",
)


args = parser.parse_args()

logger.remove()
logger.add(
    sink=f"{args.stream}_scrape_worker_{CD_TIMER[0]}.log",
    level=args.verbose * 10,
    format="<b><c><{time}</c></b> [{name}] <level>{level.name}</level> > {message}",
)

if args.file and args.table:
    if args.file.endswith(".csv"):
        file = args.file
    else:
        raise AttributeError(f"File type must be csv (end with .csv), got {args.file}")
    table = dbu.DBTable(conn, args.table)
elif args.file:
    if args.file.endswith(".csv"):
        file = args.file
    else:
        raise AttributeError(f"File type must be csv (end with .csv), got {args.file}")
    table = None
elif args.table:
    file = None
    table = dbu.DBTable(conn, args.table)
else:
    raise AttributeError(
        "Must provide stream sink for scrape worker by providing a file (csv) or database table"
        " name from connection parameters in config. Did not receive a file or table sink."
    )

if args.keywords is not None and args.keyword_fields is None:
    raise AttributeError("Keywords were provided for filtering the stream without specifying the keyword fields")

if args.stream == "reddit_comments" or args.stream == "reddit_submissions":
    if args.subreddit is not None:
        func = ru.stream_subreddit_comments if args.stream == "reddit_comments" else ru.stream_subreddit_submissions
        stream_func = partial(func, subreddit=args.subreddit)
    else:
        raise AttributeError(f"{args.stream} provided as a stream without specifying a subreddit")
elif args.stream == "finviz_news" or args.stream == "finviz_ratings":
    if args.tickers is not None:
        func = fu.stream_ticker_news if args.stream == "finviz_news" else fu.stream_ticker_ratings
        stream_func = partial(func, ticker=args.tickers, update=args.update)
    else:
        raise AttributeError(f"{args.stream} provided as a stream without specifying tickers")
elif args.stream == "motley_earnings" or args.stream == "motley_articles":
    # TODO: Add 'motley_articles' stream
    if args.stream == "motley_articles":
        raise AttributeError("Motley articles stream is not yet supported.")
    stream_func = partial(mu.stream_earnings_transcripts, update=args.update)
else:
    raise AttributeError(f"Provided stream not recognized, got: {args.stream}. Expected one of: {STREAMS}")

while True:
    try:
        stream_func(data_fields=args.fields, file=args.file, table=table)
    except Exception as err:
        print(err)
        logger.exception(f"Something horrible has happened! Error thrown from scrape worker\n{err}")
        CD_TIMER.pop(0)
        CD_TIMER.append(time.time())
        if CD_TIMER[0] - CD_TIMER[-1] < CD_LIMIT:
            logger.exception(
                f"Scrape worker failed {CD_ATTEMPTS} times in less than {CD_LIMIT} seconds! Sleeping for "
                f"{CD_TIMEOUT} seconds"
            )
            time.sleep(CD_TIMEOUT)
