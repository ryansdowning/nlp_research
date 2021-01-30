"""This module is provides various utilities for apply NLP tagging services to dataframes of text"""
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from flashtext import KeywordProcessor  # pylint: disable=E0401
from tqdm import tqdm
from transformers import PretrainedConfig  # pylint: disable=E0401
from transformers import PreTrainedTokenizer, pipeline

tqdm.pandas()
torch.backends.cudnn.enabled = True

TASK_SETTINGS = {
    "feature-extraction": {"suffix": "_extracted"},
    "sentiment-analysis": {"suffix": "_sentiment"},
    "ner": {"suffix": "_nertag"},
    "question-answering": {"suffix": "_qa"},
    "fill-mask": {"suffix": "_fill"},
    "summarization": {"suffix": "_summarization"},
    "translation_en_to_fr": {"suffix": "_en_to_fr"},
    "translation_en_to_de": {"suffix": "_en_to_de"},
    "translation_en_to_ro": {"suffix": "_en_to_ro"},
    "text-generation": {"suffix": "_generation"},
    "text2text-generation": {"suffix": "_txt2txt_generation"},
    "zero-shot-classification": {"suffix": "_zero_shot"},
    "conversation": {"suffix": "_conversation"},
    "keywords": {"suffix": "_keywords"},
}
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
SENTIMENT_SUMMARY = {"sum", "count"}


def tagging_pipeline(
    task: str,
    model: Optional = None,
    config: Optional[Union[str, PretrainedConfig]] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    framework: Optional[str] = None,
    revision: Optional[str] = None,
    use_fast: bool = True,
    **kwargs,
) -> "TaggingPipeline":
    """Utility factory method to build a tagging pipeline.

    A TaggingPipeline is a pipeline provided by transformers with its __call__ method overriden.
    The overrided functionality allows the pipeline to process dataframes with source str columns.
    Instead of outputting the standard prediction results, a call to any of these pipes will attach
    another column with the output from the pipe applied along the source columns

        Pipeline are made of:

            - A Tokenizer instance in charge of mapping raw textual input to token
            - A Model instance
            - Some (optional) post processing for enhancing model's output


    Args:
        task (`str`):
            The task defining which pipeline will be returned. Currently accepted tasks are:
            - "feature-extraction": will return child of `~transformers.FeatureExtractionPipeline`
            - "sentiment-analysis": will return child of `~transformers.TextClassificationPipeline`
            - "ner": will return child of `~transformers.TokenClassificationPipeline`
            - "question-answering": will return child of `~transformers.QuestionAnsweringPipeline`
            - "fill-mask": will return child of `~transformers.FillMaskPipeline`
            - "summarization": will return child of `~transformers.SummarizationPipeline`
            - "translation_xx_to_yy": will return a `~transformers.TranslationPipeline`
            - "text-generation": will return child of `~transformers.TextGenerationPipeline`
        model (`str` or `~transformers.PreTrainedModel` or `~transformers.TFPreTrainedModel`,
            `optional`, defaults to `None`):
            The model that will be used by the pipeline to make predictions. This can be `None`,
            a model identifier or an actual pre-trained model inheriting from
            `~transformers.PreTrainedModel` for PyTorch and `~transformers.TFPreTrainedModel` for
            TensorFlow.

            If `None`, the default for this pipeline will be loaded.
        config (`str` or `~transformers.PretrainedConfig`, `optional`, defaults to `None`):
            The configuration that will be used by the pipeline to instantiate the model.
            This can be `None`, a model identifier or an actual pre-trained model configuration
            inheriting from `~transformers.PretrainedConfig`.

            If `None`, the default for this pipeline will be loaded.
        tokenizer (`str` or `~transformers.PreTrainedTokenizer`, `optional`, defaults to `None`):
            The tokenizer that will be used by the pipeline to encode data for the model. This can
            be `None`, a model identifier or an actual pre-trained tokenizer inheriting from
            `~transformers.PreTrainedTokenizer`.

            If `None`, the default for this pipeline will be loaded.
        framework (str):
            The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified
            framework must be installed.

            If no framework is specified, will default to the one currently installed. If no
            framework is specified and both frameworks are installed, will default to PyTorch.
         revision(str):
            When passing a task name or a string model identifier: The specific model version to use. It can be a
            branch name, a tag name, or a commit id, since we use a git-based system for storing models and other
            artifacts on huggingface.co, so ``revision`` can be any identifier allowed by git.

        use_fast (bool):
            Whether or not to use a Fast tokenizer if possible (a :class:`~transformers.PreTrainedTokenizerFast`).

     Returns:
        `~tagging_utils.TaggingPipeline`: Class inheriting from `~transformers.Pipeline`, according
         to the task.
         Also provides additional functionality for pipelines such as keywords
    """

    task = task.casefold()
    # Handle special cases (not transformers pipelines)
    if task == "keywords":
        return apply_keywords_tags

    # Otherwise, get pipeline from transformers
    task_class = pipeline(
        task=task,
        model=model,
        config=config,
        tokenizer=tokenizer,
        framework=framework,
        revision=revision,
        use_fast=use_fast,
        **kwargs,
    )

    class TaggingPipeline(type(task_class)):  # pylint: disable=R0903
        """Essentially a decorator implementation of transformer's pipelines that allows for
        processing of dataframes of text elements

        Also provides functionality to implement non-transformers tagging services, such as keywords
        """

        def __call__(  # pylint: disable=W1113
            self,
            df_or_file: Union[str, pd.DataFrame],
            source: Union[str, List[str]] = "text",
            tag_suffix: Optional[str] = None,
            file: Optional[str] = None,
            *args,
            **kwargs_,
        ):
            _backend_state = torch.backends.cudnn.benchmark
            torch.backends.cudnn.benchmark = True

            tag_suffix = tag_suffix or TASK_SETTINGS[task]["suffix"]
            data, source = _handle_data_and_source(df_or_file, source)
            # Iterate through source columns and apply parent pipeline to each element with
            # the given args and kwargs
            for col in source:
                tmp = data[col].progress_apply(
                    lambda text: super(TaggingPipeline, self).__call__(text, *args, **kwargs_)
                )
                data.loc[:, f"{col}{tag_suffix}"] = tmp

            torch.backends.cudnn.benchmark = _backend_state

            # If a file is given, save to file as well
            if file is not None:
                data.to_csv(file)
            return data

    # If the device was converted to torch, default it back to -1
    if task_class.device is not None:
        if isinstance(task_class.device, int):
            device = task_class.device
        else:
            device = -1
    else:
        device = -1

    params = {
        "model": task_class.model,
        "tokenizer": task_class.tokenizer,
        "modelcard": task_class.modelcard,
        "framework": task_class.framework,
        "device": device,
        "task": task,
    }
    if task not in ("feature-extraction", "fill-mask"):
        params["binary_output"] = task_class.binary_output
    # Return TaggingPipeline with the processed arguments from the task class
    return TaggingPipeline(**params)


def _handle_data_and_source(
    df_or_file: Union[str, pd.DataFrame], source: Union[str, List[str]]
) -> Tuple[pd.DataFrame, List[str]]:
    """Helper function to handle data and source inputs for common functions

    Args:
        df_or_file: Pandas dataframe or path to csv file with data
        source: column name or names containing the source text

    Returns:
        Pandas dataframe of data and source columns in a list
    """
    if isinstance(df_or_file, str):
        data = pd.read_csv(df_or_file)
    else:
        data = df_or_file

    if isinstance(source, str):
        source = [source]

    if not all(col in data.columns for col in source):
        raise AttributeError("Not all source columns were found in the given dataset.")
    return data, source


def apply_keywords_tags(  # pylint: disable=R0913
    df_or_file: Union[str, pd.DataFrame],
    keywords: Iterable[str],
    source: Union[str, List[str]] = "text",
    case_sensitive: bool = True,
    span: bool = False,
    tag_suffix: str = TASK_SETTINGS["keywords"]["suffix"],
    file: Optional[str] = None,
) -> pd.DataFrame:
    """Inserts keyword column to given dataframe which is of type: List[str] of the keywords
    found in the [source] column

    Args:
        df_or_file: Pandas dataframe or path to csv file with data
        keywords: Iterable of strings to search for in text
        source: column name or names containing the source text
        case_sensitive: Toggle keyword case sensitivity
        span: If true, will also return the spans the keywords were found in the source
        tag_suffix: Name of new column will be formatted as [source][tag_suffix]
        file: If provided, the output will be saved to a given file path as csv

    Returns:
        Pandas dataframe of data with attached keywords column
    """
    data, source = _handle_data_and_source(df_or_file, source)

    # Get keyword processor and add keywords
    proc = KeywordProcessor(case_sensitive=case_sensitive)
    proc.add_keywords_from_list(keywords)

    # Extract keywords from all elements of each source column
    for col in source:
        data[f"{col}{tag_suffix}"] = data[col].progress_apply(lambda sent: proc.extract_keywords(sent, span_info=span))

    # If file is provided, save to file as well
    if file is not None:
        data.to_csv(file)
    return data


def filter_keywords(
    data: pd.DataFrame, keywords: Iterable[str], source: Union[str, List[str]] = "text", case_sensitive: bool = True
) -> pd.DataFrame:
    """Filters out rows that do not have any keywords in any source column(s)

    Args:
        data: dataframe containing [source] column(s) of type str
        keywords: Iterable of strings to search for in text
        source: column name or names containing the source text
        case_sensitive: Toggle keyword case sensitivity

    Returns:
        Original dataframe with rows filtered out
    """
    # Get keyword processor and add keywords
    proc = KeywordProcessor(case_sensitive=case_sensitive)
    proc.add_keywords_from_list(keywords)
    # If single source column, only need to check one element in each row, otherwise, apply any(..)
    # to check all source columns iteratively through each row
    if isinstance(source, str):
        mask = data[source].apply(lambda sent: bool(proc.extract_keywords(sent)))
    else:
        mask = data[source].apply(lambda sents: any(bool(proc.extract_keywords(sent)) for sent in sents))
    output = data[mask]  # Use mask to filter out rows without any keywords
    return output


def _multi_hot_encode(labels):
    """Helper function to transform List[List[Any]] to List[List[int]] with 1-to-1 multi-hot
    encoding across ordered unique items of original 2d list.

    Args:
        labels: 2d list/array/matrix of values

    Returns:
        Takes the ordered unique values of labels (flattened) and assigns (0, 1) values
        corresponding to whether the value is present at the same index of [labels]

        The return shape is num_unique_labels x len(labels)
    """
    # Flatten labels and get unique values (ordered)
    unique_labels = pd.Series(np.unique(np.concatenate(labels, axis=0)))

    def _get_encoding(label):
        """Helper function to transform label to multi-hot from unique values vector

        Args:
            label: Iterable containing 0 or more values
            unique: List of unique values to create vector from

        Returns:
             Series with same length as [unique], each position containing a 1 if value was found
             in [label] and 0 otherwise
        """
        return unique_labels.isin(label).astype(int)

    encoded = pd.Series(labels).apply(_get_encoding)
    encoded.columns = unique_labels
    return encoded


def get_keywords_sentiment(
    df_or_file: Union[pd.DataFrame, str],
    keywords: List[str],
    source: str = "text",
    case_sensitive: bool = True,
    model_name_or_path: str = SENTIMENT_MODEL,
    summarize: str = "sum",
) -> pd.Series:
    """Calculates the difference in the number of positively and negatively predicted keywords found
    throughout the given source texts

    Args:
        df_or_file: Pandas dataframe or path to csv file with data
        keywords: Iterable of strings to search for in text
        source: column name or names containing the source text
        case_sensitive: Toggle keyword case sensitivity
        model_name_or_path: name of transformers model to use or path to a model on local machine

    Returns:
        Series containing the sentiment scores for the corresponding keywords in same order as keywords input
    """
    if (summarize := summarize.casefold()) not in SENTIMENT_SUMMARY:
        raise AttributeError(f"Summarize currently supports: {SENTIMENT_SUMMARY}, got: {summarize}")
    # Remove rows without any keywords
    data = apply_keywords_tags(df_or_file, keywords, source, case_sensitive)
    if not any(data[f"{source}{TASK_SETTINGS['keywords']['suffix']}"].astype(bool)):
        raise ValueError("No keywords found in source text")
    data = data[data[f"{source}{TASK_SETTINGS['keywords']['suffix']}"].astype(bool)]

    # Get sentiment with (-1, 1) labels
    sentiment_tagger = tagging_pipeline("sentiment-analysis", model_name_or_path)
    data = sentiment_tagger(data, source)

    def _label_to_score(label):
        """Helper function to transform sentiment output to [-1, 1] values respective to [NEGATIVE, POSITIVE]"""
        try:
            return 1 if label[0]["label"] == "POSITIVE" else -1
        except TypeError:
            return 0

    # Convert label to int
    data.loc[:, "sentiment_score"] = data[f"{source}{TASK_SETTINGS['sentiment-analysis']['suffix']}"].apply(
        _label_to_score
    )

    # Encode keyword vectors and multiply them against the sentiment score to get keyword scores
    encoded = _multi_hot_encode(data[f"{source}{TASK_SETTINGS['keywords']['suffix']}"].values)
    keyword_scores = encoded.T * data["sentiment_score"].values

    default = None
    if summarize == "sum":
        keyword_scores = keyword_scores.sum(axis=1)
        default = 0
    elif summarize == "count":

        def _sentiment_tuple(score_row):
            return (score_row == 1).sum(), (score_row == -1).sum()

        keyword_scores = keyword_scores.apply(_sentiment_tuple, axis=1)
        default = (0, 0)

    # Fill keywords without any matches to a sentiment of 0
    missing = list(set(keywords) - set(keyword_scores.index))
    missing_scores = pd.Series([default] * len(missing), index=missing)
    keyword_scores = pd.concat((keyword_scores, missing_scores))
    return keyword_scores[keywords]  # Sort according to input


def get_keywords_occurrences(
    df_or_file: Union[pd.DataFrame, str], keywords: List[str], source: str = "text", case_sensitive: bool = True
):
    """Counts the occurrences of each keyword found throughout the given source texts

    Args:
        df_or_file: Pandas dataframe or path to csv file with data
        keywords: Iterable of strings to search for in text
        source: column name or names containing the source text
        case_sensitive: Toggle keyword case sensitivity

    Returns:
        Series containing the number of occurrences of the given keywords
        Index is same order as keywords input
    """
    # Get data with keywords
    data = apply_keywords_tags(df_or_file, keywords, source, case_sensitive)
    if not any(data[f"{source}{TASK_SETTINGS['keywords']['suffix']}"].astype(bool)):
        raise ValueError("No keywords found in source text")
    data = data[data[f"{source}{TASK_SETTINGS['keywords']['suffix']}"].astype(bool)]

    # encode the keyword vectors and sum them along the axis
    encoded = _multi_hot_encode(data[f"{source}{TASK_SETTINGS['keywords']['suffix']}"].values)
    occurrences = encoded.sum()

    # Fill keywords without any matches to 0 occurrences
    missing = list(set(keywords) - set(occurrences.index))
    occurrences = pd.concat((occurrences, pd.Series(0, index=missing)))
    return occurrences[keywords]  # Sort according to input


def keyword_bool_proc(keywords, case_sensitive):
    """Creates a process that returns a boolean value that corresponds to whether or not the passed in text contains
    any of the provided keywords (respective to case sensitivity)

    Args:
        keywords: Iterable of strings to search for in text
        case_sensitive: Toggle keyword case sensitivity

    Returns:
        Callable(str) -> bool that returns True if the string contains a keyword and false otherwise
    """
    proc = KeywordProcessor(case_sensitive=case_sensitive)
    proc.add_keywords_from_list(keywords)

    def bool_proc(text):
        return bool(proc.extract_keywords(text))

    return bool_proc
