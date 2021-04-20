"""This script provides a CLI for a general data stream from a scraping API/function to a SQL table, csv file, or both
the stream also provides additional functionality that is useful for the sentiment package such as timestamping and
real-time sentiment inference
"""
import argparse
import time
import traceback
from functools import partial

from loguru import logger

from finlang.config.db_config import conn
from finlang.nlp_utils import db_utils as dbu
from finlang.scrape_utils import finviz_utils as fu
from finlang.scrape_utils import motley_utils as mu
from finlang.scrape_utils import reddit_utils as ru

# Keeps the stream from spamming an API target or bad code from repeatedly erroring out
# This will not ever terminate the worker, the user should monitor their logs to ensure it is running stably
CD_TIMEOUT = 60
CD_LIMIT = 10
CD_ATTEMPTS = 3
CD_TIMER = [time.time()] * CD_ATTEMPTS
STREAMS = ("reddit_comments", "reddit_submissions", "finviz_news", "finviz_ratings", "motley_earnings", "motley_news")

parser = argparse.ArgumentParser()

# BASE STREAM ARGUMENTS
parser.add_argument(
    "-s", "--stream", type=str, help=f"Stream name for scraping data, currently supports: {STREAMS}", choices=STREAMS
)
parser.add_argument(
    "--fields",
    nargs="*",
    type=str,
    help="Which fields from the stream to include in the output stream. Default None, all fields are included. "
    "Must be a subset of the given streams fields.",
)
parser.add_argument("-f", "--file", default=None, type=str, help="Path of output file, must be csv")
parser.add_argument(
    "-t",
    "--table",
    default=None,
    type=str,
    help="Name of table in database to write data to. Will use the connection parameters specified in"
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
    "--timestamp_field",
    type=str,
    default='_timestamp',
    help="If --timestamp is used, the timestamp_field may be provided to specify the column name of the timestamp in "
         "the stream sink"
)
parser.add_argument(
    "--output_fields",
    nargs='*',
    type=str,
    help='Optional argument, if supplied, must be the same length as data_fields. This overwrites the output column'
         'names for the data fields for csv and database columns'
)
parser.add_argument(
    "--meta_cols",
    nargs='*',
    type=str,
    help='If provided, must also provide metadata. These are static columns to be added to each entry of the data'
         ' stream from the scrape worker'
)
parser.add_argument(
    "--metadata",
    nargs='*',
    type=str,
    help='Must be the same length as meta_cols. This is the corresponding data to add to each entry in the data stream,'
         ' for each respective meta col provided'
)
parser.add_argument(
    "--sentiment_model",
    type=str,
    help='If provided, will apply sentiment model of given name/path to the sentiment_source column and output to the '
         'sentiment_dest column. If sentiment_source is not provided, the first data_field will be assumed. If '
         'sentiment_dest is not provided, the model name will be used as output column'
)
parser.add_argument(
    "--sentiment_source",
    type=str,
    help='...'
)
parser.add_argument(
    "--sentiment_dest",
    type=str,
    help='...'
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

# Get output stream arguments for file and table if provided
if args.file and args.table:
    if args.file.endswith(".csv"):
        FILE = args.file
    else:
        raise AttributeError(f"File type must be csv (end with .csv), got {args.file}")
    TABLE = dbu.DBTable(conn, args.table)
elif args.file:
    if args.file.endswith(".csv"):
        FILE = args.file
    else:
        raise AttributeError(f"File type must be csv (end with .csv), got {args.file}")
    TABLE = None
elif args.table:
    FILE = None
    TABLE = dbu.DBTable(conn, args.table)
else:
    raise AttributeError(
        "Must provide stream sink for scrape worker by providing a file (csv) or database table"
        " name from connection parameters in config. Did not receive a file or table sink."
    )

# Validate keywords and keyword_fields arguments if provided
if args.keywords is not None and args.keyword_fields is None:
    raise AttributeError("Keywords were provided for filtering the stream without specifying the keyword fields")

# Create field map if different output fields are given
if args.output_fields is not None:
    if len(args.output_fields) != len(args.fields):
        raise AttributeError("If providing alternative output fields, the length must match the given data fields")
    FIELD_MAP = dict(zip(args.fields, args.output_fields))
else:
    FIELD_MAP = None


# Add metadata arguments as necessary
if args.meta_cols and args.metadata:
    if not len(args.meta_cols) == len(args.metadata):
        raise AttributeError("When providing meta_cols and metadata, the lengths of each must match")
    METADATA = dict(zip(args.meta_cols, args.metadata))
elif args.meta_cols is not None and args.metadata is None:
    raise AttributeError("If providing meta_cols, metadata must also be provided")
elif args.metadata is not None and args.meta_cols is None:
    raise AttributeError("If providing metadata, meta_cols must also be provided")
else:
    METADATA = None


#  Get proper sentiment model parameters if given
if args.sentiment_model is not None:
    if args.sentiment_source is not None:
        SENTIMENT_SOURCE = args.sentiment_source
    else:
        SENTIMENT_SOURCE = args.fields[0]
    if args.sentiment_dest is not None:
        SENTIMENT_DEST = args.sentiment_dest
    else:
        SENTIMENT_DEST = args.sentiment_model
else:
    SENTIMENT_SOURCE = None
    SENTIMENT_DEST = None

# Select the appropriate data stream function
if args.stream == "reddit_comments" or args.stream == "reddit_submissions":
    if args.subreddit is not None:
        func = ru.stream_subreddit_comments if args.stream == "reddit_comments" else ru.stream_subreddit_submissions
        stream_func = partial(func, subreddit=args.subreddit)
    else:
        raise AttributeError(f"{args.stream} provided as a stream without specifying a subreddit")
elif args.stream == "finviz_news" or args.stream == "finviz_ratings":
    if args.tickers is not None:
        func = fu.stream_ticker_news if args.stream == "finviz_news" else fu.stream_ticker_ratings
        tickers = [ticker.upper() for ticker in args.tickers]
        stream_func = partial(func, ticker=tickers, update=args.update)
    else:
        raise AttributeError(f"{args.stream} provided as a stream without specifying tickers")
elif args.stream == "motley_earnings" or args.stream == "motley_articles":
    # TODO: Add 'motley_articles' stream
    if args.stream == "motley_articles":
        raise AttributeError("Motley articles stream is not yet supported.")
    stream_func = partial(mu.stream_earnings_transcripts, update=args.update)
else:
    raise AttributeError(f"Provided stream not recognized, got: {args.stream}. Expected one of: {STREAMS}")

# Start data streaming to outputs, if errors are encountered, they will be logged. A cooldown is implemented to prevent
# erroneous data stream from continuously spamming logs. The settings can be found at the top of the module, but
# essentially if a data stream errors out many times in a short period of time, the stream will not be restarted until
# a cooldown period is waited
while True:
    try:
        # Start data stream
        stream_func(
            data_fields=args.fields,
            file=FILE,
            table=TABLE,
            keywords=args.keywords,
            keyword_fields=args.keyword_fields,
            case_sensitive=args.case_sensitive,
            timestamp=args.timestamp,
            timestamp_field=args.timestamp_field,
            output_field_map=FIELD_MAP,
            metadata=METADATA,
            sentiment_model=args.sentiment_model,
            sentiment_source=SENTIMENT_SOURCE,
            sentiment_dest=SENTIMENT_DEST,
        )
    except Exception as err:
        # Catch error and invoke cooldown if errors were thrown repeatedly
        logger.exception(
            f"Something horrible has happened! Error thrown from scrape worker\n{err}\n{traceback.format_exc()}"
        )
        # Remove oldest error timestamp and add current err timestamp (cycle the list)
        CD_TIMER.pop(0)
        CD_TIMER.append(time.time())
        # If the time between the oldest and most recent error timestamps is less than the limit, invoke a cooldown
        # and throw an additional exception
        if CD_TIMER[0] - CD_TIMER[-1] < CD_LIMIT:
            logger.exception(
                f"Scrape worker failed {CD_ATTEMPTS} times in less than {CD_LIMIT} seconds! Sleeping for "
                f"{CD_TIMEOUT} seconds"
            )
            time.sleep(CD_TIMEOUT)
