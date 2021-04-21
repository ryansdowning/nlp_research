import pytest
import pandas as pd


@pytest.fixture(name='text_df')
def create_text_df():
    return pd.DataFrame(
        {
            'text': [
                "This is a test sentence",
                "Here is a positive happy sentence that has good keywords which will likely have great sentiment score",
                "A sentence with a bad sentiment score has scary dark words and ugly language included"
            ]
        }
    )