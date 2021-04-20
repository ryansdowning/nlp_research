"""This module provides utilities for interacting with SQL databases, specifically mssql. Devises an OOP framework for
interacting with database tables and implements a basic factory for generating WHERE clauses for sql queries. Also
provides necessary API functionality for the alpha sentiment dashboard
"""
import datetime
from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import Enum
from functools import partial
from typing import (Any, Callable, Dict, Iterable, List, Optional, Set, Tuple,
                    Union)

import dask.dataframe as dd
import pandas as pd
import pyodbc
from loguru import logger
from tqdm import trange

from finlang.config.db_config import conn
from finlang.nlp_utils import tagging_utils as tu

Record = Union[Dict[str, Any], Tuple[Any]]
Value = Any


class Operation(ABC):
    """Base class for the different possible operations used in SQL where conditions"""
    @abstractmethod
    def __init__(self):
        pass

    @staticmethod
    def format_value(value: Value):
        """Format value for where clause"""
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, (str, datetime.datetime, datetime.date)):
            return f"'{value}'"
        raise ValueError(f"Value type not recognized! Currently does not support formatting type: {type(value)}")

    @abstractmethod
    def format_clause(self):
        """Create conditional clause for the respective operation"""
        raise NotImplementedError


class SingleOp(Operation):
    """Implements the conditionals that interact with a single operation such as greater than, not equal to, etc."""
    SINGLE_OPS = {'lt': '<', 'gt': '>', 'le': '<=', 'ge': '>=', 'eq': '=', 'ne': '<>'}

    def __init__(self, val: Value, *, op: str):
        if (op := op.casefold()) not in SingleOp.SINGLE_OPS:
            raise AttributeError(f"Operation not supported: {op}, must be one of {SingleOp.SINGLE_OPS.values()}")
        self.op = op
        self.val = val

    def format_clause(self):
        return f"{SingleOp.SINGLE_OPS[self.op]} {Operation.format_value(self.val)}"


class Between(Operation):
    """Implements BETWEEN condition"""
    def __init__(self, lower: Value, upper: Value):
        self.lower = Operation.format_value(lower)
        self.upper = Operation.format_value(upper)

    def format_clause(self):
        return f"BETWEEN {self.lower} AND {self.upper}"


class In(Operation):
    """Implements IN condition"""
    def __init__(self, vals: Iterable[Value]):
        self.vals = [Operation.format_value(val) for val in vals]

    def format_clause(self):
        in_clause = ', '.join(f"{val}" for val in self.vals)
        return f"IN ({in_clause})"


class Like(Operation):
    """Implements LIKE condition for matching regex"""
    def __init__(self, pattern: str):
        self.pattern = pattern

    def format_clause(self):
        return f"LIKE '{self.pattern}'"


class Operator(Enum):
    """Create Enum for the different conditions that can be used in the WHERE clause factory"""
    LT = partial(SingleOp, op='lt')
    GT = partial(SingleOp, op='gt')
    LE = partial(SingleOp, op='le')
    GE = partial(SingleOp, op='ge')
    EQ = partial(SingleOp, op='eq')
    NE = partial(SingleOp, op='ne')
    BETWEEN = Between
    IN = In
    LIKE = Like


# Define what a clause consists of
Expression = Tuple[Operator, Any]
Rule = Tuple[str, Expression]
Clause = Iterable[Rule]


def construct_where(clause: Clause, conditional: str = 'AND') -> str:
    """Factory for generating SQL where clauses for the given clause data structure

    Args:
        clause: Iterable of rules to include in the WHERE clause
        conditional: Operator to join the rules on. Default "AND"

    Returns:
        String of the resulting WHERE clause from the provided rules/conditions
    """
    rules = []
    for column, (op, value) in clause:
        if op == Operator.BETWEEN:
            operation = op.value(*value)
        else:
            operation = op.value(value)
        rules.append(f"{column} {operation.format_clause()}")
    rules = f" {conditional} ".join(rules)
    return f"WHERE {rules}"


class DBTable:
    """
    Class for interacting with an SQL table within a given database
    """

    def __init__(self, connection: pyodbc.Connection, table_name: str, load: bool = False):
        """
        Args:
            connection: pyodbc connection to sql database
            table_name: name of table to connect to in database
            load: If true, will load the full table into memory on initialization, in attribute `self.data`
                  Default False, does not load data
        """
        self.conn = connection
        self.table = table_name
        self.primary_keys, self.table_types, self.table_info, self.columns = None, None, None, None
        self._update_table_metadata()

        if load:
            self.data = self.select()
        else:
            self.data = None

    def _update_table_metadata(self):
        """Gets basic information about the SQL table including primary keys, columns, types, and info"""
        cursor = self.conn.cursor()
        self.primary_keys = tuple(row[3] for row in cursor.primaryKeys(table=self.table))
        cursor.execute(f"SELECT * FROM {self.table}")
        self.table_types = OrderedDict((col[0], col[1]) for col in cursor.description)
        self.table_info = OrderedDict((col[0], col[2:]) for col in cursor.description)
        self.columns = set(self.table_types.keys())
        cursor.close()

    def load_data(self):
        """
        Load all data from table into `self.data` attribute
        """
        self.data = self.select()

    def execute(
            self, query, params: Optional[Tuple[Any]] = None, commit: bool = False, many: bool = False
    ) -> pd.DataFrame:
        """Execute arbitrary query within the respective database

        WARNING: this is NOT limited to just the provided table, executing SQL commands through this method can access
        anything on the database just like a normal SQL command

        Args:
            query: The entire SQL query (can contain placeholders)
            params: Values for each placeholder in the query
            commit: Whether to commit the changes to the database
            many: Whether to use executemany when executing the query with params

        Returns:
            Dataframe of the fetched result, if the query is an insert without a return this will be an empty dataframe
        """
        logger.info("Executing query: %s" % query)
        cursor = self.conn.cursor()
        execute = cursor.executemany if many else cursor.execute
        if params is not None:
            execute(query, params)
        else:
            execute(query)

        # Check for null response
        if cursor.description is None:
            if not commit:
                raise ValueError(
                    "The query returned no data and the commit flag was not enabled. Are you sure this was a valid"
                    " query? Did you mean to commit these changes?"
                )
            return pd.DataFrame()

        data = cursor.fetchall()
        output_cols = [col[0] for col in cursor.description]

        if commit:
            conn.commit()
        cursor.close()

        if data is None:
            if not commit:
                raise ValueError(
                    "The query returned no data and the commit flag was not enabled. Are you sure this was a valid"
                    " query? Did you mean to commit these changes?"
                )
            return pd.DataFrame()
        return pd.DataFrame(dict(zip(output_cols, zip(*data))))

    def _execute_commit(self, query: str, params: Optional[Iterable[Any]] = None, many: bool = False):
        """Helper function to save duplicate code on basic SQL inserts and commits

        Args:
            query: insertion query
            params: placeholder params, if any. Default None, will not pass any placeholder values
            many: Whether to use executemany when executing the query with params
        """

        cursor = self.conn.cursor()
        execute = cursor.executemany if many else cursor.execute
        if params is not None:
            execute(query, params)
        else:
            execute(query)
        cursor.commit()
        cursor.close()

    def select(
            self, columns: Optional[Iterable[str]] = None, where_clause: Optional[Union[str, Clause]] = None
    ) -> pd.DataFrame:
        """Execute a select query on the table

        Args:
            columns: list of columns to select from table
            where_clause: string, where condition to filter rows from database. If None, no effect.
                          Also supports passing Clause, an iterable of rules which can be used to construct a where
                          clause with the construct_where function API. See db_utils.construct_where for more info
        Returns:
            Dataframe of the fetched result
        """
        if columns is None:
            columns = ["*"]
        elif not set(columns).issubset(self.columns):  # check columns exist in table
            missing = self.columns - set(columns)
            raise AttributeError(
                f"Columns attribute contains columns not available in this data table.\nMissing: {missing}"
            )

        # Add where query if necessary
        if where_clause is None:
            where_query = ""
        elif isinstance(where_clause, str):
            where_query = f" {where_clause}"
        elif isinstance(where_clause, (list, tuple, Iterable)):
            where_query = " " + construct_where(where_clause)
        else:
            raise AttributeError(
                f"Invalid where clause provided, expected String or Clause, got type: {type(where_clause)}"
            )
        query = f"SELECT {','.join(columns)} FROM {self.table}{where_query}"

        logger.info("Executing query: %s" % query)
        cursor = self.conn.cursor()
        cursor.execute(query)
        data = cursor.fetchall()
        cursor.close()

        if not data:
            logger.error("Last query failed to return any data!")
            raise ValueError("Query did not return any data, are you sure this is a valid query?")

        output_cols = list(self.table_types.keys()) if columns == ["*"] else columns
        return pd.DataFrame(dict(zip(output_cols, zip(*data))))

    def insert_one(self, record: Record, columns: Optional[Iterable[str]] = None) -> bool:
        """Executes insertion to table

        Args:
            record: Dictionary of {column: value} pairs or Tuple of values.
                    If tuple, columns must also be specified, unless using all columns in default order
                    If dictionary, columns will be ignored
            columns: ordered list of columns for insertion order

        Returns:
            True if successfully inserted
        """
        base_query = f"INSERT INTO {self.table} "
        # convert to list of column, value tuples
        record = list(self.get_standard_record(record, columns).items())
        columns = tuple(k for k, _ in record)
        vals = tuple(v for _, v in record)

        # Create insertion query
        query = f"{base_query}({','.join(columns)}) VALUES ({('?, '*len(vals))[:-2]})"
        logger.info("Executing query: %s with vals: %s" % (query, vals))
        self._execute_commit(query, vals)
        return True

    def insert_df(self, data: pd.DataFrame, chunksize: Optional[int] = None) -> bool:
        """Executes command to insert dataframe into table

        Args:
            data: data to be inserted into table. Columns must be subset of table columns
            chunksize: how many rows to be inserted at a time. If None, entire dataframe will be inserted

        Returns:
            True if successful
        """
        columns = data.columns
        if not set(columns).issubset(self.columns):
            missing = self.columns - set(columns)
            raise AttributeError(
                f"Columns attribute contains columns not available in this data table.\nMissing: {missing}"
            )
        # If chunksize is specified, handle it simply by recursively calling this function without the chunksize
        # parameter, with each chunk of data
        if chunksize is not None:
            chunk_iter = list(range(0, data.shape[0], chunksize))
            n = len(chunk_iter)
            for i, idx in enumerate(chunk_iter):
                logger.info("Inserting chunk %d of %d" % (i, n))
                self.insert_df(data.iloc[idx: idx + chunksize])
        else:
            vals = data.values.tolist()
            value_query = ",".join("?" for _ in range(len(columns)))
            query = f"INSERT INTO {self.table} ({','.join(columns)}) VALUES ({value_query})"
            logger.info("Executing many query: %s\nData has length: %d" % (query, len(vals)))
            self._execute_commit(query, vals, many=True)
        return True

    def insert_many(
        self,
        records: Iterable[Record],
        columns: Optional[Iterable[str]] = None,
        chunksize: Optional[int] = None,
        default: Optional[Any] = None,
    ) -> bool:
        """Executes command to insert many records into SQL table

        Args:
            records: Iterable of: Dictionary of {column: value} pairs or Tuple of values.
                     If tuple, columns must also be specified, unless using all columns in default order
                     If dictionary, columns will be ignored
            columns: ordered list of columns for insertion order
            chunksize: How many records to insert per query. Defaults to None, all records are inserted at once
            default: What to replace NaN and None-like data with. Defaults to None

        Returns:
            Returns true if the insert was completed successfully
        """
        # If chunksize is specified, handle it simply by recursively calling this function without the chunksize
        # parameter, with each chunk of data. Chunks are gathered by iterating until chunksize or end of iterable is
        # reached, whichever is first.
        if chunksize is not None:
            _chunk_row = 0
            while records is not None:
                chunk = []
                for _ in range(chunksize):
                    try:
                        chunk.append(next(records))
                    except StopIteration:
                        records = None
                        break
                logger.info("Inserting chunk: rows %d to %d", (_chunk_row, _chunk_row + len(chunk)))
                _chunk_row += len(chunk)
                self.insert_many(records=chunk, columns=columns, default=default)
        else:
            records = [self.get_standard_record(record, columns) for record in records]
            # Convert to dataframe to put into column
            data = pd.DataFrame.from_records(records).fillna(default)
            self.insert_df(data)
        return True

    def add_column(self, column: str, data_type: str) -> bool:
        """Create a new column in the SQL table and update the tables metadata

        Args:
            column: Name of new column to add to the table
            data_type: SQL data type of the new column

        Raises:
            AttributeError: If the column already exists in the given table

        Returns:
            Returns True if column was added successfully
        """
        if column in self.columns:
            raise AttributeError(f"The column: {column} is already being used in table: {self.table}")
        query = f"""ALTER TABLE {self.table}\nADD \"{column}\" {data_type}"""
        logger.info(f"Adding columns: {column} with type: {data_type} to table: {self.table}\nExecuting query: {query}")
        self._execute_commit(query)
        self._update_table_metadata()
        return True

    def update_column_on_primary(self, column: str, new_val: Any, primary_key: Tuple) -> bool:
        """Updates a single entry in a given column to a new value

        Args:
            column: Column in table to update
            new_val: Value to set the column to
            primary_key: Primary key corresponding to the row in the table to update

        Raises:
            ValueError: If primary keys for table were not found
            KeyError: If the given primary key tuple length does not equal the number of primary key columns in
                      the table

        Returns:
            Returns True if column entry was updated successfully
        """
        if not self.primary_keys:
            raise ValueError(
                "Failed to find primary keys when initializing data table, please add them manually by accessing the "
                "primary_keys attribute if you need to use this function."
            )
        if len(self.primary_keys) != len(primary_key):
            raise KeyError(
                f"Primary key supplied must be a tuple of values for each primary key column. Got non-matching length "
                f"tuple provided. The primary keys for this column are currently set as: {self.primary_keys}"
            )

        query = f"UPDATE {self.table} SET \"{column}\" = ? " \
                f"WHERE {' AND '.join(f'{pk_col} = ?' for pk_col in self.primary_keys)}"
        params = (new_val,) + primary_key
        logger.info(
            f"Updating column: {column} on primary key: {self.primary_keys} = {primary_key}"
            f" to value: {new_val} in table: {self.table}"
        )
        self._execute_commit(query, params)
        return True

    def update_column_on_multiple_primary_keys(
            self, column: str, new_vals: List[Any], primary_keys: List[Tuple]
    ) -> bool:
        """Updates a single column with multiple new values corresponding to rows matching the provided primary keys

        Args:
            column: Column in table to update
            new_vals: List of new values to set in the column, one for each row to be updated
            primary_keys: List of primary keys representing the rows to update, must be the same length as `new_vals`

        Raises:
            ValueError: If primary keys for table were not found
            KeyError: If the given primary key tuple length does not equal the number of primary key columns in
                      the table

        Returns:
            Returns True if all rows in the column were updated successfully
        """
        if not self.primary_keys:
            raise ValueError(
                "Failed to find primary keys when initializing data table, please add them manually by accessing the "
                "primary_keys attribute if you need to use this function."
            )
        if len(new_vals) != len(primary_keys):
            raise AttributeError(
                f"Length of new values to update and the list of primary key tuples must be equal, got {len(new_vals)} "
                f"new values, and {len(primary_keys)} primary keys."
            )
        if not all(len(self.primary_keys) == len(pk) for pk in primary_keys):
            raise KeyError(
                f"Primary keys supplied must be a list of tuples of values for each primary key column. Got unequal"
                f" length tuple provided. The primary keys for this column are currently set as: {self.primary_keys}"
            )

        query = f"""UPDATE {self.table} SET \"{column}\" = ?
WHERE {' AND '.join(f'{pk_col} = ?' for pk_col in self.primary_keys)}"""
        params = [(val, *pkey) for val, pkey in zip(new_vals, primary_keys)]
        logger.info(
            f"Updating column: {column} on {len(primary_keys)} different primary keys with {len(new_vals)} new values "
            f"in table: {self.table}"
        )
        self._execute_commit(query, params, many=True)
        return True

    def check_types(self, data: Iterable[Any], columns: Iterable[str]):
        """Ensures the data provided matches the data type specifications of the corresponding columns

        Args:
            data: Iterable of elements
            columns: Iterable of column names corresponding to each data element

        Raises:
            AttributeError: If any of the columns provided are not present in the given table
            ValueError: If any data element has an unexpected type, not matching with the table's column specification

        Returns:
            True if the data is valid
        """
        if columns and not set(columns).issubset(self.columns):
            missing = self.columns - set(columns)
            raise AttributeError(
                f"Columns attribute contains columns not available in this data table.\nMissing: {missing}"
            )
        for item, col in zip(data, columns):
            if not isinstance(item, self.table_types[col]):
                raise ValueError(
                    f"Data for column {col} incorrect. Got {item} of type {type(item)},"
                    f" expected {self.table_types[col]}"
                )
        return True

    def get_standard_record(self, record: Record, columns: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        """Helper function that takes a record which has a general structure and converts it to a standard record, or
        a dictionary of column: value pairs

        Args:
            record: Non-standard record. Can be a tuple of values or a dictionary (already standard). If tuple of values
                    is passed and the length does not match the number of columns in the table, the columns parameter
                    must be specified.
            columns: Columns corresponding to the given values in the record, when the record is a tuple

        Returns:
            Dictionary of column: value pairs, standard record
        """
        if isinstance(record, tuple):
            if not columns and len(record) != len(self.columns):
                raise AttributeError(
                    f"If the length of the data tuple is not equal to the number of data fields in the table, "
                    f"columns must be provided. Got {len(record)} elements, {self.table} has {len(self.columns)}"
                    f" columns"
                )
            if columns and len(columns) != len(record):
                raise AttributeError(
                    f"If columns are provided, they must match the length of the data tuple given. Got {len(columns)}"
                    f" columns and {len(record)} data elements"
                )
            if columns and not set(columns).issubset(set(self.columns)):
                missing = set(self.columns) - set(columns)
                raise AttributeError(
                    f"Columns attribute contains columns not available in this data table.\nMissing: {missing}"
                )
            columns = set(columns) if columns else self.columns
            return dict(zip(columns, record))
        if isinstance(record, dict):
            if not set(record.keys()).issubset(self.columns):
                missing = set(record.keys()) - self.columns
                raise AttributeError(
                    f"The keys of the data dictionary must only contain columns in this data table. Missing: {missing}"
                )
            if columns:
                logger.warning(
                    "Columns were provided for inserting data dictionary. Ignoring columns and using dictionary keys."
                    " Call using tuple if columns must be specified."
                )
            return record
        raise AttributeError(f"Record must be of type tuple or dictionary, got: {type(record)}")


def _dt_to_ts(date):
    """Helper function to convert datetime objects to integers because for some reason this isn't a builtin option"""
    return int(datetime.datetime.timestamp(date))


def get_sentiment_data(
        model: str,
        ids: Optional[List[int]] = None,
        start: Optional[datetime.datetime] = None,
        end: Optional[datetime.datetime] = None,
        sources: Optional[List[str]] = None,
        cols: Optional[Iterable[str]] = ('id',)
) -> pd.DataFrame:
    """Full API wrapper for the sentiment SQL table, providing optional conditional queries on each column

    Args:
        model: Name of sentiment model (corresponding to column name) to pull sentiment data for
        ids: Optionally filter on a list of ids (rows)
        start: Optionally provide a start date to filter rows on, only select rows >= start
        end: Optionally provide an end date to filter rows on, only select rows <= end
        sources: Optionally provide a list of sources to filter on, only select rows WHERE source IN sources
        cols: List of columns, in addition to the model column to return. By default only the id column will be included

    Returns:
        Dataframe of sentiment data from the built SQL query
    """
    sentiment = DBTable(conn, 'sentiment')
    if model not in sentiment.columns:
        avail_models = sentiment.columns - {'id', 'created_utc', 'source', 'text'}
        raise AttributeError(
            f"The model: {model} was not found in the sentiment table.\n"
            f"Please choose from the list of available models are: {avail_models}"
        )
    cols = set(cols)
    cols.add(model)
    cols.add('id')

    where_rules = []
    if ids is not None:
        where_rules.append(('id', (Operator.IN, ids)))
    if start is not None and end is not None:
        where_rules.append(('created_utc', (Operator.BETWEEN, (_dt_to_ts(start), _dt_to_ts(end)))))
    elif start is not None:
        where_rules.append(('created_utc', (Operator.GE, _dt_to_ts(start))))
    elif end is not None:
        where_rules.append(('created_utc', (Operator.LE, _dt_to_ts(end))))
    if sources is not None:
        where_rules.append(('source', (Operator.IN, sources)))
    where_cond = construct_where(where_rules) if where_rules else None

    data = sentiment.select(cols, where_cond)
    if 'created_utc' in data.columns:  # Convert date column from timestamp to datetime if returning created_utc
        data['created_utc'] = pd.to_datetime(data['created_utc'], unit='s')
    return data


def get_sentiment_linking(ids: Optional[List[int]] = None, keywords: Optional[List[str]] = None) -> pd.DataFrame:
    """API for keyword-sentiment linking table

    Args:
        ids: If provided, query will be filtered to only include rows WHERE id IN ids
        keywords: If provided, query will be filtered to only include rows WHERE keyword IN keywords

    Returns:
        Dataframe of ids and keywords which represent that the id in the sentiment table contains the respective keyword
    """
    linking = DBTable(conn, 'keyword_linking')

    where_rules = []
    if ids is not None:
        where_rules.append(('id', (Operator.IN, ids)))
    if keywords is not None:
        where_rules.append(('keyword', (Operator.IN, keywords)))
    where_cond = construct_where(where_rules) if where_rules else None

    data = linking.select(('id', 'keyword'), where_cond)
    data['id'] = data['id'].astype(int)
    return data


def get_sentiment_by_keywords(
        model: Union[str, List[str]],
        keywords: Optional[List[str]] = None,
        start: Optional[datetime.datetime] = None,
        end: Optional[datetime.datetime] = None,
        sources: Optional[List[str]] = None,
        cols: Optional[Iterable[str]] = ('id',),
        return_linking: Optional[bool] = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Set[int]]]]:
    """API that connects the sentiment table and keyword-linking table for fast querying on sentiment data, filtered
    on a select number of keywords

    Args:
        model: Model(s) to pull sentiment data for
        keywords: List of keywords to pull sentiment data for. Default None, all data will be pulled
        start: Optionally provide a start date to filter rows on, only select rows >= start
        end: Optionally provide an end date to filter rows on, only select rows <= end
        sources: Optionally provide a list of sources to filter on, only select rows WHERE source IN sources
        cols: List of columns, in addition to the model column to return. By default only the id column will be included
        return_linking: If true, the function will also return a dictionary of the keyword: List[ids] for the selected
                        keywords. Default False, only the final sentiment dataframe is returned

    Returns:
        Returns the dataframe of sentiment data for the respective query, if return_linking is True, a dictionary of the
        keyword-ids will be returned as well (return will be a tuple of dataframe, dictionary)
    """
    sentiment = DBTable(conn, 'sentiment')
    # Gather list of columns needed for query
    cols = set(cols)
    cols.add('id')
    if isinstance(model, str):
        cols.add(model)
    elif isinstance(model, list):
        cols.update(model)
    # Convert to strings and add linking columns
    columns = [f"{sentiment.table}.\"{col}\"" for col in cols]
    if return_linking:
        columns += ["keyword_linking.\"id\"", "keyword_linking.\"keyword\""]
    columns = ', '.join(columns)

    # Create keyword filtering condition
    keyword_query = ""
    if keywords is not None:
        keyword_query = f"WHERE keyword_linking.keyword IN " \
                        f"({', '.join(Operation.format_value(val) for val in keywords)})"
    query = f"""SELECT {columns} FROM {sentiment.table}
     INNER JOIN keyword_linking ON keyword_linking.id = {sentiment.table}.id 
     {keyword_query}"""

    # Add other query conditions
    where_rules = []
    if start is not None and end is not None:
        where_rules.append(('created_utc', (Operator.BETWEEN, (_dt_to_ts(start), _dt_to_ts(end)))))
    elif start is not None:
        where_rules.append(('created_utc', (Operator.GE, _dt_to_ts(start))))
    elif end is not None:
        where_rules.append(('created_utc', (Operator.LE, _dt_to_ts(end))))
    if sources is not None:
        where_rules.append(('source', (Operator.IN, sources)))
    where_cond = construct_where(where_rules) if where_rules else None

    # Combine conditionals
    if where_cond is not None:
        if keyword_query:
            query += f" AND {where_cond[6:]}"
        else:
            query += where_cond

    sentiment_data = sentiment.execute(query)
    # Convert date column from timestamp to datetime if returning created_utc
    if 'created_utc' in sentiment_data.columns:
        sentiment_data['created_utc'] = pd.to_datetime(sentiment_data['created_utc'], unit='s')

    sentiment_data.drop_duplicates(subset=['id'], inplace=True)  # TODO: is this needed?
    if return_linking:
        # Convert linking table to a dictionary
        linking = sentiment_data[['id', 'keyword']].groupby('keyword')['id'].apply(set).to_dict()
        sentiment_data = sentiment_data[list(cols)]
        return sentiment_data, linking
    return sentiment_data


def sentiment_inference_pipeline(
        model: [str, Callable[[str], int]],
        table: DBTable,
        source_col: str,
        dest_col: str,
        chunk_size: Optional[int] = None,
        fill_null: Optional[bool] = False,
        workers: Optional[int] = None,
        verbose: Optional[bool] = True
) -> bool:
    """Uses a model to perform sentiment prediction on a source column and insert those predictions to a destination
    column of a SQL table

    Args:
        model: Model to use for sentiment inference. If a string, huggingface transformers model will be used for
               binary [-1, 1] prediction, see `nlp_utils.tagging_utils.sentiment_sign` for more information. Otherwise,
               a callable must be provided which takes one parameter, a string, and returns an integer.
        table: DBTable object to interact with SQL table
        source_col: Name of source column in SQL table. Must be a text column to perform sentiment classification on
        dest_col: Name of destination column in SQL table. If doesn't already exist, column will be added. If it does
                  exist, column must be integer type
        chunk_size: Optionally provide a chunk size to split up the prediction insertions. If chunk size is None,
                    inference is done on all rows and then inserted all at once. Otherwise inference is done on N rows
                    at a time and are inserted in those chunks
        fill_null:  Optionally toggle fill_null to skip rows in the database where the destination column is already
                    filled (is not NULL). Useful for continuing a previous inference job that did not complete.
        workers: Optionally provide a number of workers to use. Default None, process will be done sequentially. When
                 provided, dask will be used to split the dataframe into partitions to perform inference in parallel.
        verbose:  If True, will log errors when the model throws an error during inference, otherwise errors pass
                  silently

    Raises:
        ValueError: If primary keys are not found in the given table

    Returns:
        Returns True if sentiment inference and insertion completed successfully
    """
    if not table.primary_keys:
        raise ValueError(
            f"In order to infer sentiment for each text, the primary key(s) must be set in the table. The primary"
            f" key(s) for table: {table.name} are missing or failed to be fetched."
        )
    if dest_col not in table.columns:  # Add destination column if it doesn't exist already
        table.add_column(dest_col, 'INT')
    else:
        assert table.table_types[dest_col] == int, f"Destination column: {dest_col} already exists and is not of " \
                                                   f"type int! Column must be integer type for sentiment inference"
    if isinstance(model, str):
        if workers is not None:
            model = tu.sentiment_sign(model, use_fast=False)
        else:
            model = tu.sentiment_sign(model)
    elif not callable(model):
        raise AttributeError(
            f"Model must either be a name or path to a sentiment model, or a callable that takes text as"
            f" argument and returns an int. Got: {model}"
        )

    clause = None
    if fill_null:  # If only filling null, then we only need data where destination column is null
        clause = f"WHERE \"{dest_col}\" IS NULL"
    text_data = table.select(columns=table.primary_keys + (source_col,), where_clause=clause)

    def _insert_inference(row: pd.Series) -> Tuple[Optional[int], Tuple[Any]]:
        """Processes sentiment inference for given row

        Args:
            row: row of data containing primary key column(s) and the source column to do inference on

        Returns:
            Tuple of the sentiment prediction and primary key of the row
        """
        try:
            prediction = model(row[source_col])
        except Exception as err:  # Naively catch errors and insert None for the inference, log error if verbose is True
            if verbose:
                logger.error(err)
            prediction = None
        key = tuple(row[pkey] for pkey in table.primary_keys)
        return prediction, key

    def _process_chunk(chunk: pd.DataFrame):
        """Helper function to process sentiment inference for the given chunk of data

        Args:
            chunk: Dataframe containing source column and primary key columns to perform inference across

        Returns:
            None, inserts sentiment predictions into SQL table's destination column
        """
        if workers is not None:
            # If using workers, divide the chunk into partitions with dask and apply the inference function
            # across each partition
            data = dd.from_pandas(chunk, npartitions=workers)
            preds_and_keys = data.map_partitions(
                lambda partition: partition.apply(_insert_inference, axis=1)
            ).compute().tolist()
        else:
            # Otherwise just apply inference function normally
            preds_and_keys = chunk.apply(_insert_inference, axis=1).tolist()
        preds, keys = zip(*preds_and_keys)
        table.update_column_on_multiple_primary_keys(dest_col, preds, keys)

    if chunk_size is not None:
        # Divide dataframe into chunks of <chunk_size> rows and process each chunk individually
        for i in trange(0, text_data.shape[0], chunk_size):
            chunk = text_data.iloc[i:i+chunk_size, :]
            _process_chunk(chunk)
    else:
        # Process the entire dataframe at once
        _process_chunk(text_data)
    return True


def apply_new_keyword_tags(
        keywords: Iterable[str], case_sensitive: bool = True, start_max: bool = True, drop_duplicates: bool = True
) -> pd.DataFrame:
    """Fill the keyword linking table by performing keyword matching for rows in the sentiment table

    Args:
        keywords: Keywords to match on
        case_sensitive: Whether or not the keyword match should be case sensitive. Default True
        start_max: Default True, will look at the max id in the linking table, and continue keyword matching from that
                   id onward. Useful for continuing keyword matching that did not complete, or after adding more rows
                   to the sentiment table
        drop_duplicates: Default True, will pull the current linking table and drop the duplicates before inserting to
                         ensure errors are not thrown from re-adding a keyword match

    Returns:
        Returns the linking data that was inserted into the linking table
    """
    sentiment = DBTable(conn, 'sentiment')
    linking = DBTable(conn, 'keyword_linking')
    if start_max:
        # Get max id in the linking table
        max_id = int(linking.execute(f"SELECT MAX(id) FROM {linking.table}").iloc[0, 0])
        clause = [('id', (Operator.GT, max_id))]
    else:
        clause = None
    text_data = sentiment.select(['id', 'text'], where_clause=clause)

    # Perform keyword matching
    keyword_proc = tu.tagging_pipeline('keywords')
    text_data = keyword_proc(text_data, source='text', keywords=list(keywords), case_sensitive=case_sensitive)
    # Expand linking index and convert to pandas dataframe for insertion
    linking_data = tu.expanded_list_linking_index(
        text_data, keyword_col=f"text{tu.TASK_SETTINGS['keywords']['suffix']}", index_col="id"
    )
    linking_data = [(id_, keyword) for keyword, ids in linking_data.items() for id_ in ids]
    linking_data = pd.DataFrame(linking_data, columns=['id', 'keyword']).sort_values('id')

    if drop_duplicates:  # Drop duplicates if necessary
        curr_linking = linking.select(('keyword', 'id'))
        linking_data = linking_data[
            ~(linking_data['id'].isin(curr_linking['id']) & linking_data['keyword'].isin(curr_linking['keyword']))
        ]
    # Insert and return new rows
    linking.insert_df(linking_data)
    return linking_data
