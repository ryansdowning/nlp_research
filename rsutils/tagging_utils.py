from typing import Iterable, List, Optional, Union, Tuple

import pandas as pd

from flashtext import KeywordProcessor
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _handle_data_and_source(
        df_or_file: Union[str, pd.DataFrame], source: Union[str, List[str]]
) -> Tuple[pd.DataFrame, List[str]]:
    if isinstance(df_or_file, str):
        data = pd.read_csv(df_or_file)
    else:
        data = df_or_file

    if isinstance(source, str):
        source = [source]
    else:
        source = source

    if not all(col in data.columns for col in source):
        raise AttributeError("Not all source columns were found in the given dataset.")
    return data, source


def apply_sentiment_tags(
        df_or_file: Union[str, pd.DataFrame],
        model_name_or_path: str,
        source: Union[str, List[str]] = 'text',
        tag_suffix: str = '_sentiment',
        file: Optional[str] = None,
):
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    data, source = _handle_data_and_source(df_or_file, source)

    for col in source:
        data[f"{col}{tag_suffix}"] = data[col].apply(lambda x: get_sentiment(x, model, tokenizer))

    if file is not None:
        data.to_csv(file)
    return data


def get_sentiment(
        text: str,
        model: AutoModelForSequenceClassification,
        tokenizer: AutoTokenizer,
        score: bool = False,
) -> Union[float, int]:
    features = tokenizer.encode_plus(
        text, max_length=512, pad_to_max_length=True, truncation=True, return_tensors='pt'
    )
    result, *_ = model(**features)
    return float(result) if score else int(result.argmax())


def apply_keywords_tags(
        df_or_file: Union[str, pd.DataFrame],
        keywords: Iterable[str],
        source: Union[str, List[str]] = 'text',
        case_sensitive: bool = False,
        span: bool = False,
        tag_suffix: str = '_keywords',
        file: Optional[str] = None,
):
    data, source = _handle_data_and_source(df_or_file, source)

    proc = KeywordProcessor(case_sensitive=case_sensitive)
    proc.add_keywords_from_list(keywords)

    for col in source:
        data[f"{col}{tag_suffix}"] = data[col].apply(
            lambda sent: proc.extract_keywords(sent, span_info=span)
        )

    if file is not None:
        data.to_csv(file)
    return data


def filter_keywords(
        data: pd.DataFrame,
        keywords: Iterable[str],
        source: Union[str, List[str]] = 'text',
        case_sensitive: bool = False
) -> pd.DataFrame:
    proc = KeywordProcessor(case_sensitive=case_sensitive)
    proc.add_keywords_from_list(keywords)
    if isinstance(source, str):
        mask = data[source].apply(lambda sent: bool(proc.extract_keywords(sent)))
    else:
        mask = data[source].apply(
            lambda sents: any(bool(proc.extract_keywords(sent)) for sent in sents)
        )
    output = data[mask]
    return output
