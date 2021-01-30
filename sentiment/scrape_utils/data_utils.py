import csv
import os
import re
import time
from typing import Any, Generator, Iterable, List, Optional, Tuple, Union

import bs4
from loguru import logger

from sentiment.nlp_utils import db_utils as dbu
from sentiment.nlp_utils import tagging_utils as tu


def open_file_stream(file: str, data_fields: List[str]):
    """Prepares file for streaming data into

    Args:
        file: Path of file to stream data to
        data_fields: Ordered list of attributes included in the data stream

    Returns:
        returns 0 if preparing the file was successful, produces an error otherwise
    """
    logger.info("Checking if file exists at: %s" % file)
    if os.path.exists(file):
        with open(file, "rt", encoding="utf-8") as out:
            reader = csv.reader(out)
            fields = next(reader)
        if fields != data_fields:
            raise OSError(
                f"File already exists at: {file}\nDoes not share the same data fields so data "
                f"cannot be appended. Please specify a different file path or match the existing"
                f" data fields\nCurrent fields: {fields}"
            )
    else:
        with open(file, "w") as out:
            csv_out = csv.writer(out)
            csv_out.writerow(data_fields)
    return 0


def process_attr(obj, field):
    """Helper function to extract attributes and nested attributes from objects

    Args:
        obj: object that contains [field] as attribute
        field: attribute of object

    Returns:
        Attribute of object corresponding to the respective field
    """
    out = obj
    for attr in field.split("."):
        out = getattr(out, attr)
    return out


def get_attributes_list(obj: Any, data_fields: Iterable[str]) -> Tuple[Any]:
    """Gets the values of the given attributes from the object

    Args:
        obj: Object to get values from - must have all attributes in data fields parameter
        data_fields: The attributes to extract from the object

    Returns:
        List of values extracted from the object, respective to the given data fields (in order)
    """
    return tuple(map(lambda field: process_attr(obj, field), data_fields))


def get_data_fields(data: Iterable[Any], fields: Iterable[str]):
    """Gets a list of attributes for [n] of objects of the same type

    Args:
        data: Iterable of objects containing [fields]
        fields: attributes to extract from each data object

    Returns:
        List of tuples containing the [fields] from each object in data
        Size will be len(data) x len(fields)
    """
    return [get_attributes_list(point, fields) for point in data]


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
        if hasattr(elem, "name"):
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
    return re.sub(r"\s{2,}", "\n", "\n".join(page))


def stream_data(
    stream: Generator[Any, None, None],
    data_fields: List[str] = None,
    file: Optional[str] = None,
    table: Optional[dbu.DBTable] = None,
    keywords: Optional[List[str]] = None,
    keyword_fields: Optional[str] = None,
    case_sensitive: bool = True,
    timestamp: bool = False,
):
    """Creates a process that indefinitely streams data from a subreddit to a file

    Args:
        stream: Generator that outputs a stream of data corresponding to the given data fields
        data_fields: Ordered list of attributes to extract from submissions/comments
        keywords: If None, this argument has no affect. If keywords are provided, comments and submissions are filtered
                  if they do not contain a keyword in their body or selftext respectively
        keyword_fields: String corresponding to the data field(s) to match keywords for. If multiple are provided, the
                       fields will be joined by a space before searching for keywords.
        case_sensitive: If keywords are provided, flag for case sensitive matching. Default True, on.
        file: Path to file (csv) to write data to
        table: DBTable object that can be provided to insert data directly into SQL database
        timestamp: If true, will add the timestamp the data was saved as an entry in the row. Default False

    Returns:
        None - Process runs until error is thrown or is interrupted
        Data stream is written to the given file or SQL table
    """
    if not file and not table:
        raise AttributeError("Must provide at least one of <file> or <table> parameters to stream data to!")
    if keywords is not None and keyword_fields is None:
        raise AttributeError("Keywords were provided without specifying any keyword fields to match to!")

    if keywords:
        if isinstance(keyword_fields, str):
            keyword_fields = {keyword_fields}

        keyword_fields = set(keyword_fields)
        try:
            idxs = [data_fields.index(field) for field in keyword_fields]
        except ValueError:
            raise AttributeError(
                f"Could not find all keyword fields in the provided data fields!\n"
                f"Missing: {keyword_fields - set(data_fields)}"
            )
        proc = tu.keyword_bool_proc(keywords=keywords, case_sensitive=case_sensitive)
    else:
        idxs = None
        proc = None

    if file:
        open_file_stream(file, data_fields)
    cols = ", ".join(data_fields)
    for data in stream:
        if proc is not None:  # If filtering for keywords, run keyword check on keyword fields
            keyword_data = " ".join(str(data[idx]) for idx in idxs)
            if not proc(keyword_data):  # If no match, skip this iteration
                continue
        if timestamp:
            data += (time.time(),)
        if table:
            table.insert(data, data_fields)
        if file:
            with open(file, "a", encoding="utf-8") as out:
                logger.info("writing %s data to to file at: %s" % (cols, file))
                csv_out = csv.writer(out)
                csv_out.writerow(data)
