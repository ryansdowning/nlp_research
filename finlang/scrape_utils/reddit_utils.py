import datetime
import os
from typing import Any, Iterable, Iterator, List, Optional, Tuple, Union

import pandas as pd
import praw
from loguru import logger
from psaw import PushshiftAPI
from tqdm import tqdm

from finlang.config.constants import COMMENT_FIELDS, SUBMISSION_FIELDS, SUBMISSION_SORTS, reddit
from finlang.nlp_utils import db_utils as dbu
from finlang.scrape_utils import data_utils as du

api = PushshiftAPI()


def get_submissions(subreddit: praw.reddit.Subreddit, sort_method: str, **kwargs) -> Iterator[Any]:
    """Gets the submissions from a specified subreddit using a given sorting option33

    Args:
        subreddit: Name of subreddit
        sort_method: Method of sorting to use when pulling the submissions
        kwargs: additional keyword arguments used in the specific sort method call

    Returns:
        List of submissions retrieved from the subreddit
    """
    sort_method = sort_method.casefold()
    if sort_method not in SUBMISSION_SORTS:
        raise AttributeError(
            f"Unexpected sort method: {sort_method}\nSupported sort methods " f" for submissions: {SUBMISSION_SORTS}"
        )

    logger.info("Attempting to get submissions from subreddit: %s, using sort method of: %s" % (subreddit, sort_method))
    if sort_method == "hot":
        return subreddit.hot(**kwargs)
    elif sort_method == "new":
        return subreddit.hot(**kwargs)
    elif sort_method == "rising":
        return subreddit.rising(**kwargs)
    elif sort_method == "controversial":
        return subreddit.controversial(**kwargs)
    elif sort_method == "top":
        return subreddit.top(**kwargs)
    elif sort_method == "gilded":
        return subreddit.gilded(**kwargs)

    raise NotImplementedError(f"Getting submissions with sort method of: {sort_method} has not yet been implemented.")


def filter_marked(
    data: Iterable[praw.reddit.Submission], marked_labels: Iterable[str], field: str = "id"
) -> List[praw.reddit.Submission]:
    """Helper function to remove marked elements from list of elements using given field

    Args:
        data: Iterable of objects with attribute corresponding to field
        marked_labels: Iterable of <field> values from the marked submissions
        field: The attribute to check for submissions

    Returns:
        List of submissions with the marked submissions removed
    """
    if field not in set(SUBMISSION_FIELDS + COMMENT_FIELDS):
        raise AttributeError(
            f"{field} is not a valid submission attribute, please choose from:"
            f" {set(SUBMISSION_FIELDS + COMMENT_FIELDS)}"
        )
    return [entity for entity in data if getattr(entity, field) not in marked_labels]


def get_submissions_data(submissions: Iterable[praw.reddit.Submission], data_fields: Iterable[str]) -> List[Tuple]:
    """Helper function to retrieve the data fields from a list of submissions

    Args:
        submissions: List of submissions to extract data from
        data_fields: Ordered list of attributes to extract from submissions

    Returns:
        List of tuples containing the data fields (ordered as given in input) from each submission
    """
    if not set(data_fields).issubset(SUBMISSION_FIELDS):
        raise AttributeError(
            f"Unexpected data fields for subreddit submissions:" f" {set(data_fields) - set(SUBMISSION_FIELDS)}"
        )

    data = du.get_data_fields(submissions, data_fields)
    return data


def get_comments_data(comments: Iterable[praw.reddit.Comment], data_fields: Iterable[str]) -> List[Tuple]:
    """Helper function to retrieve the data fields from a list of comments

    Args:
        comments: List of comments to extract data from
        data_fields: Ordered list of attributes to extract from submissions

    Returns:
        List of tuples containing the data fields (ordered as given in input) from each submission
    """
    if not set(data_fields).issubset(COMMENT_FIELDS):
        raise AttributeError(
            f"Unexpected data fields for subreddit comments:" f" {set(data_fields) - set(COMMENT_FIELDS)}"
        )

    data = du.get_data_fields(comments, data_fields)
    return data


def get_subreddit_submissions(
    subreddit: str,
    sort_method: Optional[str] = "new",
    data_fields: Optional[List[str]] = None,
    file: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """Scrapes subreddit submissions and stores the data in pandas dataframe, or csv file

    Args:
        subreddit: Name of subreddit
        sort_method: Method of sorting to use when pulling the submissions
        data_fields: Ordered list of attributes to extract from submissions
        file: Default None. If specified, data will be written to this file (csv)
        kwargs: Additional keyword arguments are passed into the pull request for the given
                sort method

    Returns:
        Pandas df of submissions data - also writes df to csv if file is given
    """
    if sort_method not in SUBMISSION_SORTS:
        raise AttributeError(
            f"Unexpected sort method: {sort_method}\nSupported sort methods " f" for submissions: {SUBMISSION_SORTS}"
        )

    if data_fields is None:
        data_fields = list(SUBMISSION_FIELDS)
    elif not set(data_fields).issubset(SUBMISSION_FIELDS):
        raise AttributeError(
            f"Unexpected data fields for subreddit submissions:" f" {set(data_fields) - set(SUBMISSION_FIELDS)}"
        )

    sub = reddit.subreddit(subreddit)
    submissions = get_submissions(sub, sort_method, **kwargs)
    data = get_submissions_data(submissions, data_fields)

    data = pd.DataFrame(data, columns=data_fields)
    if file is not None:
        data.to_csv(file)
    return data


def get_submission_comments(
    submissions: Iterable[praw.reddit.Submission], limit: Optional[int] = None
) -> List[praw.reddit.Comment]:
    """Iterates through reddit submissions and extracts the comments into a 1d list of reddit comments

    Args:
        submissions: iterable of reddit submissions to get the comments from
        limit: number of comments to pull from each submission. If None, will pull all comments

    Returns:
        list of comments extracted from the submissions
    """
    comments = []
    for submission in submissions:
        logger.info("Using replace more to get additional comments for submission: %s" % submission.id)
        submission.comments.replace_more(limit=limit)
        comments.extend(submission.comments.list())
    return comments


def get_subreddit_submission_comments(
    subreddit: str,
    sort_method: str = "new",
    data_fields: Optional[List[str]] = None,
    file: Optional[str] = None,
    **kwargs,
):
    """Scrapes all of the comments from the submissions of a given subreddit, respective to the sorting method/kwargs

    Args:
        subreddit: Name of subreddit
        sort_method: Method of sorting to use when pulling the submissions
        data_fields: Ordered list of attributes to extract from submissions
        file: Default None. If specified, data will be written to this file (csv)
        kwargs: Additional keyword arguments are passed into the pull request for the given
                sort method for submissions

    Returns:
        Pandas df of comments data from scraped submissions - also writes daf to csv if file is given
    """
    if data_fields is None:
        data_fields = list(COMMENT_FIELDS)
    elif not set(data_fields).issubset(COMMENT_FIELDS):
        raise AttributeError(
            f"Unexpected data fields for subreddit comments:" f" {set(data_fields) - set(COMMENT_FIELDS)}"
        )

    sub = reddit.subreddit(subreddit)
    submissions = get_submissions(sub, sort_method, **kwargs)
    comments = get_submission_comments(submissions, None)
    data = get_comments_data(comments, data_fields)

    data = pd.DataFrame(data, columns=data_fields)
    if file is not None:
        data.to_csv(file)
    return data


def subreddit_data_stream(
    subreddit: str,
    sub_or_comm: str,
    data_fields: List[str] = None,
    **kwargs,
):
    """Creates a generator expression that yields a stream of data from a subreddit given the input parameters

    Args:
        subreddit: Name of subreddit
        sub_or_comm: identify type of data to stream: 'submissions' or 'comments'
        data_fields: Ordered list of attributes to extract from submissions/comments
        kwargs: Additional keyword arguments provided to the prawcore stream function for submissions/comments

    Returns:
        Tuples of len(data_fields) are yielded respective to the ordered data fields
    """
    if sub_or_comm not in ("submissions", "comments"):
        raise AttributeError("Must specify either submissions or comments for data stream")
    sub = reddit.subreddit(subreddit)
    data_stream = sub.stream.submissions(**kwargs) if sub_or_comm == "submissions" else sub.stream.comments(**kwargs)

    for data in data_stream:
        yield du.get_attributes_list(data, data_fields)


def stream_subreddit_submissions(
    subreddit: str,
    data_fields: Optional[List[str]] = None,
    file: Optional[str] = None,
    table: Optional[dbu.DBTable] = None,
    **kwargs,
):
    """Creates a process that indefinitely streams submissions data from a subreddit to a file

    Args:
        subreddit: Name of subreddit
        data_fields: Ordered list of attributes to extract from submissions
        file:  Name of file (csv) to write data to
        table: DBTable object that can be provided to insert data directly into SQL database
        kwargs: Additional keyword arguments for streaming data

    Returns:
        None - Process runs until error is thrown or is interrupted
        Submissions data is written to the given file
    """
    if data_fields is None:
        data_fields = list(SUBMISSION_FIELDS)
    elif not set(data_fields).issubset(SUBMISSION_FIELDS):
        raise AttributeError(
            f"Unexpected data fields for subreddit submissions:" f" {set(data_fields) - set(SUBMISSION_FIELDS)}"
        )

    stream = subreddit_data_stream(subreddit, "submissions", data_fields)
    du.stream_data(stream=stream, data_fields=data_fields, file=file, table=table, **kwargs)


def stream_subreddit_comments(
    subreddit: str,
    data_fields: Optional[List[str]] = None,
    file: Optional[str] = None,
    table: Optional[dbu.DBTable] = None,
    **kwargs,
):
    """Creates a process that indefinitely streams comments data from a subreddit to a file

    Args:
        subreddit: Name of subreddit
        data_fields: Ordered list of attributes to extract from comments
        file:  Name of file (csv) to write data to
        table: DBTable object that can be provided to insert data directly into SQL database
        kwargs: Additional keyword arguments for streaming data

    Returns:
        None - Process runs until error is thrown or is interrupted
        Comments data is written to the given file
    """
    if data_fields is None:
        data_fields = list(COMMENT_FIELDS)
    elif not set(data_fields).issubset(COMMENT_FIELDS):
        raise AttributeError(
            f"Unexpected data fields for subreddit comments:" f" {set(data_fields) - set(COMMENT_FIELDS)}"
        )

    stream = subreddit_data_stream(subreddit, "comments", data_fields)
    du.stream_data(stream, data_fields, file, table, **kwargs)


def _update_data(
    sub_or_comm: str,
    df_or_file: Union[pd.DataFrame, str],
    id_col: str = "id",
    file: Optional[str] = None,
) -> pd.DataFrame:
    """Updates reddit submissions and comments data based on their associated ids. Pulls updated data and replaces it
    in the given dataset

    Args:
        sub_or_comm: identify type of data to update: 'submissions' or 'comments'
        df_or_file: pandas df or a csv file containing old data (must have id column for updating)
        id_col: Name of column to find the ids associated with the submissions or comments. This is used to pull the
                newly updated data respective to those ids
        file: Default None. If specified, data will be written to this file (csv)

    Returns:
        Pandas df with the same columns and shape as the original dataset, but with updated values
        If file is not None, will output this data to a csv (file must be .csv)
    """
    if sub_or_comm not in ("submissions", "comments"):
        raise AttributeError("Must specify either submissions or comments for streaming data")

    if isinstance(df_or_file, pd.DataFrame):
        data = df_or_file
    elif isinstance(df_or_file, str):
        data = pd.read_csv(df_or_file)
    else:
        raise AttributeError(
            f"Invalid type: {type(df_or_file)}\ndf_or_file must be either a pandas dataframe or a"
            f" string corresponding to the path of a csv file."
        )
    del df_or_file

    fields = SUBMISSION_FIELDS if sub_or_comm == "submissions" else COMMENT_FIELDS
    if not set(data.columns).issubset(fields):
        raise AttributeError(
            f"Unexpected column(s) in dataset: {set(data.columns) - fields} for {sub_or_comm} data."
            f"\nAccepted columns for this dataset are: {fields}"
        )
    if id_col not in data.columns:
        raise AttributeError(f"id column: {id_col} does not exist in the given dataset.")

    if sub_or_comm == "submissions":

        def get_updated(id_):
            return du.get_attributes_list(reddit.submission(id_), data.columns)

    else:

        def get_updated(id_):
            return du.get_attributes_list(reddit.comment(id_), data.columns)

    data = pd.DataFrame(data[id_col].apply(get_updated).values.tolist(), columns=data.columns)
    if file is not None:
        data.to_csv(file)
    return data


def update_submissions(
    df_or_file: Union[pd.DataFrame, str], id_col: str = "id", file: Optional[str] = None
) -> pd.DataFrame:
    """Updates reddit submissions data based on their associated ids. Pulls updated data and replaces it
    in the given dataset

    Args:
        df_or_file: pandas df or a csv file containing old data (must have id column for updating)
        id_col: Name of column to find the ids associated with the submissions. This is used to pull the newly updated
                data respective to those ids
        file: Default None. If specified, data will be written to this file (csv)

    Returns:
        Pandas df with the same columns and shape as the original dataset, but with updated values
        If file is not None, will output this data to a csv (file must be .csv)
    """
    return _update_data("submissions", df_or_file, id_col, file)


def update_comments(
    df_or_file: Union[pd.DataFrame, str], id_col: str = "id", file: Optional[str] = None
) -> pd.DataFrame:
    """Updates reddit comments data based on their associated ids. Pulls updated data and replaces it
        in the given dataset

    Args:
        df_or_file: pandas df or a csv file containing old data (must have id column for updating)
        id_col: Name of column to find the ids associated with the comments. This is used to pull the newly updated
                data respective to those ids
        file: Default None. If specified, data will be written to this file (csv)

    Returns:
        Pandas df with the same columns and shape as the original dataset, but with updated values
        If file is not None, will output this data to a csv (file must be .csv)
    """
    return _update_data("comments", df_or_file, id_col, file)


def _process_pushshift_chunk(chunk: pd.DataFrame, fields: Optional[List[str]]):
    for date_col in {"created_utc", "retrieved_on", "created"}:
        if date_col in chunk.columns:
            chunk[date_col] = pd.to_datetime(chunk[date_col], unit="s")
    return chunk[fields] if fields else chunk


def _get_pushshift(
    sub_or_comm: str,
    subreddit: str,
    start: Optional[datetime.datetime] = None,
    end: Optional[datetime.datetime] = None,
    data_fields: Optional[List[str]] = None,
    file: Optional[str] = None,
    checkpoint_freq: Optional[int] = None,
    **kwargs,
) -> pd.DataFrame:
    """Function to query pushshift api using PSAW framework, implements helper functionality such as date ranges and
    checkpointing to files

    Args:
        sub_or_comm: Abstract to avoid duplicate code, submissions or comments is provided to specify exact stream type
        subreddit: Subreddit to stream from
        start: Start date(time) for data stream
        end: End date(time) for data stream
        data_fields: Columns to include in the stream
        file: File to save output to, must be provided if checkpoint_freq is specified
        checkpoint_freq: How often (integer of rows) to save/checkpoint to a file, if a file already exists at the
                         location, it will attempt to append to existing data.
        **kwargs: Additional keyword arguments provided to the PSAW search_comments or search_submissions function.
                  Please see their documentation for more details.

    Returns:

    """
    if sub_or_comm == "submissions":
        api_func = api.search_submissions
    elif sub_or_comm == "comments":
        api_func = api.search_comments
    else:
        raise AttributeError("Must specify either submissions or comments for pushshift data")

    if checkpoint_freq is not None and file is None:
        raise AttributeError("Checkpoint frequency provided without specifying a file path to save to!")

    if data_fields is not None:
        data = api_func(subreddit=subreddit, after=start, before=end, filter=data_fields, **kwargs)
    else:
        data = api_func(subreddit=subreddit, after=start, before=end, **kwargs)

    if checkpoint_freq is not None:
        items = []
        for i, item in tqdm(enumerate(data, 1)):
            items.append(item.d_)
            if i % checkpoint_freq == 0:
                data_df = _process_pushshift_chunk(pd.DataFrame(items), data_fields)
                data_df.to_csv(file, index=False, mode="a", header=not os.path.exists(file))
                items = []
        data_df = _process_pushshift_chunk(pd.DataFrame(items), data_fields)
        data_df.to_csv(file, index=False, mode="a", header=not os.path.exists(file))
        return data_df
    else:
        data_df = _process_pushshift_chunk(pd.DataFrame([item.d_ for item in tqdm(data)]), data_fields)
        if file is not None:
            data_df.to_csv(file, index=False)
        return data_df


def get_pushshift_submissions(
    subreddit: str,
    start: Optional[datetime.datetime] = None,
    end: Optional[datetime.datetime] = None,
    data_fields: Optional[List[str]] = None,
    file: Optional[str] = None,
    checkpoint_freq: Optional[int] = None,
    **kwargs,
) -> pd.DataFrame:
    """Function to stream subreddit submissions from pushshift API through PSAW worker

    Args:
        subreddit:
        subreddit: Subreddit to stream from
        start: Start date(time) for data stream
        end: End date(time) for data stream
        data_fields: Columns to include in the stream
        file: File to save output to, must be provided if checkpoint_freq is specified
        checkpoint_freq: How often (integer of rows) to save/checkpoint to a file, if a file already exists at the
                         location, it will attempt to append to existing data.
        **kwargs: Additional keyword arguments provided to the PSAW search_comments or search_submissions function.
                  Please see their documentation for more details.

    Returns:
        Dataframe of subreddit submissions for the provided query, also saves to file (csv) if provided
    """
    return _get_pushshift("submissions", subreddit, start, end, data_fields, file, checkpoint_freq, **kwargs)


def get_pushshift_comments(
    subreddit: str,
    start: Optional[datetime.datetime] = None,
    end: Optional[datetime.datetime] = None,
    data_fields: Optional[List[str]] = None,
    file: Optional[str] = None,
    checkpoint_freq: Optional[int] = None,
    **kwargs,
) -> pd.DataFrame:
    """Function to stream subreddit comments from pushshift API through PSAW worker

    Args:
        subreddit:
        subreddit: Subreddit to stream from
        start: Start date(time) for data stream
        end: End date(time) for data stream
        data_fields: Columns to include in the stream
        file: File to save output to, must be provided if checkpoint_freq is specified
        checkpoint_freq: How often (integer of rows) to save/checkpoint to a file, if a file already exists at the
                         location, it will attempt to append to existing data.
        **kwargs: Additional keyword arguments provided to the PSAW search_comments or search_submissions function.
                  Please see their documentation for more details.

    Returns:
        Dataframe of subreddit comments for the provided query, also saves to file (csv) if provided
    """
    return _get_pushshift("comments", subreddit, start, end, data_fields, file, checkpoint_freq, **kwargs)
