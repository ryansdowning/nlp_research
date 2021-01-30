from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyodbc
from loguru import logger

Record = Union[Dict[str, Any], Tuple[Any]]


class DBTable:
    """
    Class for interacting with an SQL table within a given database
    """

    def __init__(self, connection: pyodbc.Connection, table_name: str, load: bool = False):
        """
        Args:
            connection: pyodbc connection to sql database
            table_name: name of table to connect to in database
        """
        self.conn = connection
        self.table = table_name

        cursor = self.conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name}")
        self.table_types = OrderedDict((col[0], col[1]) for col in cursor.description)
        self.table_info = OrderedDict((col[0], col[2:]) for col in cursor.description)
        self.columns = set(self.table_types.keys())

        if load:
            self.data = self.select()
        else:
            self.data = None
        cursor.close()

    def load_data(self):
        """
        Load all data from table into `data` attribute
        """
        self.data = self.select()

    def select(self, columns: Optional[List[str]] = None, where_cond: Optional[str] = None):
        """Execute a select query on the table

        Args:
            columns: list of columns to select from table
            where_cond: string, where condition to filter rows from database. If None, no effect.

        Returns:

        """
        if not columns:
            columns = ["*"]
        elif not set(columns).issubset(self.columns):
            missing = self.columns - set(columns)
            raise AttributeError(
                f"Columns attribute contains columns not available in this data table.\nMissing: {missing}"
            )

        where_query = f" WHERE {where_cond}" if where_cond else ""
        query = f"SELECT {','.join(columns)} FROM {self.table}{where_query}"

        logger.info("Executing query: %s" % query)
        cursor = self.conn.cursor()
        cursor.execute(query)
        data = cursor.fetchall()
        cursor.close()

        if not data:
            logger.error("Last query failed to return any data!")
            raise ValueError("Query did not return any data, are you sure this is a valid query?")

        output_cols = self.columns if columns == ["*"] else columns
        data = np.array(data).T

        return pd.DataFrame({col: values for col, values in zip(output_cols, data)})

    def insert_one(self, record: Record, columns: Optional[Iterable[str]] = None):
        """Executes insertion to table

        Args:
            data: Dictionary of {column: value} pairs or Tuple of values.
                  If tuple, columns must also be specified, unless using all columns in default order
                  If dictionary, columns will be ignored
            columns: ordered list of columns for insertion order

        Returns:
            True if successfully inserted
        """
        base_query = f"INSERT INTO {self.table} "
        record = self.get_standard_record(record, columns)
        columns = tuple(record.keys())
        vals = tuple(record.values())

        query = f"{base_query}({','.join(columns)}) VALUES {vals}"
        cursor = self.conn.cursor()

        logger.info("Executing query: %s" % query)
        cursor.execute(query)
        cursor.commit()
        cursor.close()
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
        if chunksize is not None:
            chunk_iter = list(range(0, data.shape[0], chunksize))
            n = len(chunk_iter)
            for i, idx in enumerate(chunk_iter):
                logger.info("Inserting chunk %d of %d" % (i, n))
                self.insert_df(data.iloc[idx : idx + chunksize])
        else:
            vals = data.values.tolist()
            value_query = ",".join(f":{i}" for i in range(len(columns)))
            query = f"INSERT INTO {self.table} {','.join(columns)} VALUES ({value_query})"

            cursor = self.conn.cursor()
            logger.info("Executing many query: %s\nData has length: %d" % (query, len(vals)))
            cursor.executemany(query, vals)
            cursor.commit()
            cursor.close()

        return True

    def insert_many(
        self,
        records: Iterable[Record],
        columns: Optional[Iterable[str]] = None,
        chunksize: Optional[int] = None,
        default: Optional[Any] = None,
    ) -> bool:
        """

        Args:
            records (Iterable[Record]): [description]
            columns (Optional[Iterable[str]]): [description]. Defaults to None.
            chunksize (Optional[int]): [description]. Defaults to None.
            default (Optional[Any]): [description]. Defaults to None.

        Raises:
            ValueError: [description]

        Returns:
            bool: [description]
        """
        if chunksize is not None:
            _chunk_row = 0
            chunk = []
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
            return self.insert_df(data)

    def check_types(self, data, columns):
        """

        Args:
            data:
            columns:

        Returns:

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
            return {col: item for col, item in zip(columns, record)}
        elif isinstance(record, dict):
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
        else:
            raise AttributeError(f"Record must be of type tuple or dictionary, got: {type(record)}")
