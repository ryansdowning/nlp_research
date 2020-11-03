from loguru import logger
import os
import csv


def open_file_stream(file, data_fields):
    """Prepares file for streaming data into

    Args:
        file: Path of file to stream data to
        data_fields: Ordered list of attributes included in the data stream

    Returns:
        returns 0 if preparing the file was successful, produces an error otherwise
    """
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
    return 0
