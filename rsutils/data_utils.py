import csv
import os
from typing import Any, Iterable, Iterator, List, Optional, Tuple, Union

import pandas as pd

import praw
from loguru import logger
from rsutils.constants import (COMMENT_FIELDS, COMMENT_SORTS,
                               SUBMISSION_FIELDS, SUBMISSION_SORTS, reddit)


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
            f"Unexpected sort method: {sort_method}\nSupported sort methods "
            f" for submissions: {SUBMISSION_SORTS}"
        )

    logger.info(
        'Attempting to get submissions from subreddit: %s, using sort method of: %s'
        % (subreddit, sort_method)
    )
    if sort_method == 'hot':
        return subreddit.hot(**kwargs)
    elif sort_method == 'new':
        return subreddit.hot(**kwargs)
    elif sort_method == 'rising':
        return subreddit.rising(**kwargs)
    elif sort_method == 'controversial':
        return subreddit.controversial(**kwargs)
    elif sort_method == 'top':
        return subreddit.top(**kwargs)
    elif sort_method == 'gilded':
        return subreddit.gilded(**kwargs)

    raise NotImplementedError(
        f"Getting submissions with sort method of: {sort_method} has not yet been implemented."
    )


def filter_marked(
        data: Iterable[praw.reddit.Submission], marked_labels: Iterable[str], field: str = 'id'
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


def _get_attributes_list(obj: Any, data_fields: Iterable[str]) -> Tuple[Any]:
    # if not set(data_fields).issubset(set(obj.__dir__())):
    #     raise AttributeError(
    #         f"Unexpected data fields for object with type: {type(obj)}. Attributes not available:"
    #         f"\n{set(data_fields) - set(obj.__dir__())}"
    #     )
    return tuple(map(lambda field: getattr(obj, field), data_fields))


def _get_data_fields(data: Iterable[Any], fields: Iterable[str]):
    return [_get_attributes_list(point, fields) for point in data]


def get_submissions_data(
        submissions: Iterable[praw.reddit.Submission], data_fields: Iterable[str]
) -> List[Tuple]:
    """Helper function to retrieve the data fields from a list of submissions

    Args:
        submissions: List of submissions to extract data from
        data_fields: Ordered list of attributes to extract from submissions

    Returns:
        List of tuples containing the data fields (ordered as given in input) from each submission
    """
    if not set(data_fields).issubset(SUBMISSION_FIELDS):
        raise AttributeError(
            f"Unexpected data fields for subreddit submissions:"
            f" {set(data_fields) - set(SUBMISSION_FIELDS)}"
        )

    data = _get_data_fields(submissions, data_fields)
    return data


def get_comments_data(
        comments: Iterable[praw.reddit.Comment], data_fields: Iterable[str]
) -> List[Tuple]:
    """Helper function to retrieve the data fields from a list of comments

    Args:
        comments: List of comments to extract data from
        data_fields: Ordered list of attributes to extract from submissions

    Returns:
        List of tuples containing the data fields (ordered as given in input) from each submission
    """
    if not set(data_fields).issubset(COMMENT_FIELDS):
        raise AttributeError(
            f"Unexpected data fields for subreddit comments:"
            f" {set(data_fields) - set(COMMENT_FIELDS)}"
        )

    data = _get_data_fields(comments, data_fields)
    return data


def get_subreddit_submissions(
        subreddit: str,
        sort_method: Optional[str] = 'new',
        data_fields: Optional[List[str]] = None,
        file: Optional[str] = None,
        **kwargs,
):
    """Scrapes subreddit submissions and stores the data in pandas dataframe, or csv file

    Args:
        subreddit: Name of subreddit
        sort_method: Method of sorting to use when pulling the submissions
        data_fields: Ordered list of attributes to extract from submissions
        file: Default None. If specified, data will be written to this file (csv)
        kwargs: Additional keyword arguments are passed into the pull request for the given
                sort method
    """
    if sort_method not in SUBMISSION_SORTS:
        raise AttributeError(
            f"Unexpected sort method: {sort_method}\nSupported sort methods "
            f" for submissions: {SUBMISSION_SORTS}"
        )

    if data_fields is None:
        data_fields = list(SUBMISSION_FIELDS)
    elif not set(data_fields).issubset(SUBMISSION_FIELDS):
        raise AttributeError(
            f"Unexpected data fields for subreddit submissions:"
            f" {set(data_fields) - set(SUBMISSION_FIELDS)}"
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
    comments = []
    for submission in submissions:
        logger.info(
            'Using replace more to get additional comments for submission: %s' % submission.id
        )
        submission.comments.replace_more(limit=limit)
        comments.extend(submission.comments.list())
    return comments


def get_subreddit_submission_comments(
        subreddit: str,
        sort_method: str = 'new',
        data_fields: Optional[List[str]] = None,
        file: Optional[str] = None,
        **kwargs,
):
    if data_fields is None:
        data_fields = list(COMMENT_FIELDS)
    elif not set(data_fields).issubset(COMMENT_FIELDS):
        raise AttributeError(
            f"Unexpected data fields for subreddit comments:"
            f" {set(data_fields) - set(COMMENT_FIELDS)}"
        )

    sub = reddit.subreddit(subreddit)
    submissions = get_submissions(sub, sort_method, **kwargs)
    comments = get_submission_comments(submissions, None)
    data = get_comments_data(comments, data_fields)

    data = pd.DataFrame(data, columns=data_fields)
    if file is not None:
        data.to_csv(file)
    return data


def _stream_subreddit_data(
        subreddit: str,
        sub_or_comm: str,
        file: str,
        data_fields: List[str],
        **kwargs
):
    if sub_or_comm not in ('submissions', 'comments'):
        raise AttributeError("Must specify either submissions or comments for streaming data")
    sub = reddit.subreddit(subreddit)
    data_stream = (
        sub.stream.submissions(**kwargs)
        if sub_or_comm == 'submissions'
        else sub.stream.comments(**kwargs)
    )

    logger.info('Checking if file exists at: %s' % file)
    if os.path.exists(file):
        with open(file, 'rt', encoding='utf-8') as out:
            reader = csv.reader(out)
            fields = next(reader)
        if fields != data_fields:
            raise OSError(
                f"File already exists at: {file}\nDoes not share the same data fields so data "
                f"cannot be appended. Please specify a different file path or match the existing"
                f" data fields\nCurrent fields: {fields}"
            )
    else:
        with open(file, 'w') as out:
            csv_out = csv.writer(out)
            csv_out.writerow(data_fields)

    for data in data_stream:
        extracted = _get_attributes_list(data, data_fields)
        with open(file, 'a', encoding='utf-8') as out:
            logger.info('writing %s data to to file at: %s' % (sub_or_comm, file))
            csv_out = csv.writer(out)
            csv_out.writerow(extracted)


def stream_subreddit_submissions(
        subreddit: str, file: str, data_fields: Optional[List[str]] = None, **kwargs
):
    if data_fields is None:
        data_fields = list(SUBMISSION_FIELDS)
    elif not set(data_fields).issubset(SUBMISSION_FIELDS):
        raise AttributeError(
            f"Unexpected data fields for subreddit submissions:"
            f" {set(data_fields) - set(SUBMISSION_FIELDS)}"
        )

    _stream_subreddit_data(subreddit, 'submissions', file, data_fields, **kwargs)


def stream_subreddit_comments(
        subreddit: str, file: str, data_fields: Optional[List[str]] = None, **kwargs
):
    if data_fields is None:
        data_fields = list(COMMENT_FIELDS)
    elif not set(data_fields).issubset(COMMENT_FIELDS):
        raise AttributeError(
            f"Unexpected data fields for subreddit comments:"
            f" {set(data_fields) - set(COMMENT_FIELDS)}"
        )

    _stream_subreddit_data(subreddit, 'comments', file, data_fields, **kwargs)


def _update_data(
        sub_or_comm: str,
        df_or_file: Union[pd.DataFrame, str],
        id_col: str = 'id',
        file: Optional[str] = None,
) -> pd.DataFrame:
    if sub_or_comm not in ('submissions', 'comments'):
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

    fields = SUBMISSION_FIELDS if sub_or_comm == 'submissions' else COMMENT_FIELDS
    if not set(data.columns).issubset(fields):
        raise AttributeError(
            f"Unexpected column(s) in dataset: {set(data.columns) - fields} for {sub_or_comm} data."
            f"\nAccepted columns for this dataset are: {fields}"
        )
    if id_col not in data.columns:
        raise AttributeError(f"id column: {id_col} does not exist in the given dataset.")

    if sub_or_comm == 'submissions':

        def get_updated(id_):
            return _get_attributes_list(reddit.submission(id_), data.columns)

    else:

        def get_updated(id_):
            return _get_attributes_list(reddit.comment(id_), data.columns)

    data = pd.DataFrame(data[id_col].apply(get_updated).values.tolist(), columns=data.columns)
    if file is not None:
        data.to_csv(file)
    return data


def update_submissions(
        df_or_file: Union[pd.DataFrame, str], id_col: str = 'id', file: Optional[str] = None
) -> pd.DataFrame:
    return _update_data('submissions', df_or_file, id_col, file)


def update_comments(
        df_or_file: Union[pd.DataFrame, str], id_col: str = 'id', file: Optional[str] = None
) -> pd.DataFrame:
    return _update_data('comments', df_or_file, id_col, file)
