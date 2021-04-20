"""This module provides functionality for summarizing sentiment data and constructing visualizations based on the
summarized data"""
import datetime
from functools import reduce
from typing import Dict, List, Optional, Set, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

SOURCE_LABEL = "Source"
KEYWORD_LABEL = "Keyword"
SUMMARY_LABEL = "Summary"
DATE_LABEL = "Date"
SUMMARY_TYPES = ("mentions", "count", "sum")
POS_LABEL, NEG_LABEL = 'positive', 'negative'
COUNT_COLORS = {POS_LABEL: "green", NEG_LABEL: "red"}
PLOT_BACKENDS = ("plotly", "matplotlib")


def compute_facet_data(
    data: pd.DataFrame,
    keywords_linking: Dict[str, Set],
    date_col: str,
    source_col: str,
    label_col: Optional[str] = None,
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None,
    sources: Optional[Union[str, List[str]]] = None,
    keywords: Optional[Union[str, List[str]]] = None,
    frequency: Optional[str] = "m",
    summary: Optional[str] = "mentions",
    facet_keywords: Optional[bool] = False,
    facet_sources: Optional[bool] = False,
):
    """Computes the summary data respective to the provided facet settings and summary type. Performs data filtering
    if the optional parameters are supplied

    The data provided also has a corresponding keywords_linking that maps keywords to the respective indexes of the data
    table. This helps save a lot of memory with large datasets and also speeds up computation when working with many
    keywords

    Args:
        data: Full dataset with date_col, source_col, and (optionally) label_col. Further, the provided
              keyword_linking contains the indexes of the data where the keyword is in the data's keywords
        keywords_linking: Dictionary where the keys are keywords in the data's text and and the values are sets of
                          index values corresponding to indexes of data where the keyword is present in the data
        date_col: Name of column in data that contains the date values for plotting, must be datetime
        source_col: Name of column in data that contains the source name of the data for plotting
        label_col: [Optional] Name of column in data contains the [sentiment] label, should be [-1, 1], but the code is
                   generalized to support any ints or even floats probably. Default None, must be provided when using
                   summary type 'sum' or 'count' since those summary methods needs that label data
        start_date: start date for plotting date. Default None, if provided, the data will be filtered to only
                    include data where date_col >= start_date
        end_date: end date for plotting date. Default None, if provided, the data will be filtered to only
                  include data where date_col <= end_date
        sources: Sources to include in the plotted data. Default None, if provided the data will be filtered to only
                 include data where the source_col is one of sources. Can be single str, or list of strings.
        keywords: Keywords to include in the plotted data. Default None, if provided the data will be filtered to only
                  include data where the keywords_linking contains the index of one of the provided keywords. Can be
                  single str, or list of strings.
        frequency: How often to compute summary statistic (must be compatible with pandas groupby) default 'm', monthly.
        summary: Summary method, currently supports:
                 mentions - Size of each data group
                 sum - Sum of label_col for each data group
                 count - count of positive and negative values in label_col for each data group
        facet_keywords: Boolean flag, whether or not to divide the plotted lines among the individual keywords
                        Default False, does not facet on keywords. If True, must provide <keywords> parameter
        facet_sources: Boolean flag, whether or not to divide the plotted lines among the individual sources
                       Default False, does not facet on sources. If True, must provide <sources> parameter

    Returns:
        Dataframe of the summary data to be used for plotting
    """
    # Valid Input Checking
    summary = summary.casefold()
    if (summary := summary.casefold()) not in SUMMARY_TYPES:
        raise AttributeError(f"Got unexpected value for summary: {summary}, expected str one of: {SUMMARY_TYPES}")
    if summary in ("sum", "count") and label_col is None:
        raise AttributeError(f"Must provide name label_col for summary type: {summary}")
    if facet_keywords and keywords is None:
        keywords = list(keywords_linking.keys())
    if facet_sources and sources is None:
        sources = data[source_col].unique().tolist()

    # Filter data
    if start_date is None:
        start_filter = np.full(data.shape[0], True)
    else:
        start_filter = data[date_col] >= start_date
    if end_date is None:
        end_filter = np.full(data.shape[0], True)
    else:
        end_filter = data[date_col] <= end_date
    if sources is None:
        sources_filter = np.full(data.shape[0], True)
    else:
        if isinstance(sources, str):
            sources = [sources]
        sources_filter = data[source_col].isin(sources)
    if keywords is None:
        keywords_filter = np.full(data.shape[0], True)
    else:
        if isinstance(keywords, str):
            keywords = [keywords]
        keywords_indexes = reduce(lambda x, y: x.union(keywords_linking[y]), keywords, set())
        keywords_filter = data.index.isin(keywords_indexes)
    data_filter = start_filter & end_filter & sources_filter & keywords_filter  # Join filter
    plot_data = data[data_filter]
    plot_idxs = set(plot_data.index)  # Get new subset of indexes for keyword linking

    def _mentions(_data):
        """Computes length of data (number of mentions)"""
        mentions_data = _data.size()
        return pd.DataFrame({DATE_LABEL: mentions_data.index, summary: mentions_data.tolist()})

    def _sum(_data):
        """Computes sum of sentiment scores"""
        sum_data = _data[label_col].sum()
        return pd.DataFrame({DATE_LABEL: sum_data.index, summary: sum_data.tolist()})

    def _count(_data):
        """Computes count (as a tuple) of positive and negative sentiment scores"""
        count_data = _data[label_col].apply(lambda x: ((x > 0).sum(), (x < 0).sum()))
        count_data = pd.DataFrame(
            count_data.tolist(),
            columns=(POS_LABEL, NEG_LABEL),
            index=count_data.index,
        )
        count_data[DATE_LABEL] = count_data.index
        return count_data.reset_index()

    summary_func = eval(f"_{summary}")  # Get appropriate summarizing function, this is pretty safe use of eval
    grouper = pd.Grouper(key=date_col, freq=frequency)
    facet_data = []
    if facet_keywords and facet_sources:
        for keyword in keywords:
            idxs = keywords_linking[keyword].intersection(plot_idxs)
            for source in sources:
                data_group = plot_data.loc[idxs]
                data_group = data_group[data_group[source_col] == source].groupby(grouper)
                summary_data = summary_func(data_group)
                summary_data[KEYWORD_LABEL] = keyword
                summary_data[SOURCE_LABEL] = source
                facet_data.append(summary_data)
    elif facet_keywords:
        for keyword in keywords:
            idxs = keywords_linking[keyword].intersection(plot_idxs)
            data_group = plot_data.loc[idxs].groupby(grouper)
            summary_data = summary_func(data_group)
            summary_data[KEYWORD_LABEL] = keyword
            facet_data.append(summary_data)
    elif facet_sources:
        for source in sources:
            data_group = plot_data[plot_data[source_col] == source].groupby(grouper)
            summary_data = summary_func(data_group)
            summary_data[SOURCE_LABEL] = source
            facet_data.append(summary_data)
    else:
        facet_data.append(summary_func(plot_data.groupby(grouper)))

    facet_data = pd.concat(facet_data)
    return facet_data


def plot_keywords(
    facet_data: pd.DataFrame,
    date_col: str,
    label_col: Optional[str] = None,
    summary: Optional[str] = "mentions",
    facet_keywords: Optional[bool] = False,
    facet_sources: Optional[bool] = False,
    overlap: Optional[bool] = False,
    backend: Optional[str] = "plotly",
):
    """Dynamic plotting function that works with upstream text data that has been tagged in some (numerical) way

    The provided functionality allows the user to pass in a dataframe including a date_col, source_col, and label_col,
    and then specify various options for producing summarized graphs that represent the given data. Multiple summarizing
    methods are currently supported and the faceting optionality allows the user to visualize large data in a meaningful
    and clear manner.

    Args:
        facet_data: Full dataset with date_col, source_col, and (optionally) label_col. Further, the provided
              keyword_linking contains the indexes of the data where the keyword is in the data's keywords
        date_col: Name of column in data that contains the date values for plotting, must be datetime
        label_col: [Optional] Name of column in data contains the [sentiment] label, should be [-1, 1], but the code is
                   generalized to support any ints or even floats probably. Default None, must be provided when using
                   summary type 'sum' or 'count' since those summary methods needs that label data
        summary: Summary method, currently supports:
                 mentions - Size of each data group
                 sum - Sum of label_col for each data group
                 count - count of positive and negative values in label_col for each data group
        facet_keywords: Boolean flag, whether or not to divide the plotted lines among the individual keywords
                        Default False, does not facet on keywords. If True, must provide <keywords> parameter
        facet_sources: Boolean flag, whether or not to divide the plotted lines among the individual sources
                       Default False, does not facet on sources. If True, must provide <sources> parameter
        overlap: If the data itself is facetted among the keywords or sources, this flag allows the lines to all be
                 drawn on a single plot rather than using seaborn FacetGrid to draw one line on each subplot
                 Default False, facets plots according to facet options
        backend: Change the plotting backend used, by default the plotly package will be used. Specifying "matplotlib"
                 will instead use matplotlib backend for plotting

    Returns:
        fig with the data plotted corresponding to the given options and the selected plotting backend package
    """
    # Valid Input Checking
    summary = summary.casefold()
    if (summary := summary.casefold()) not in SUMMARY_TYPES:
        raise AttributeError(f"Got unexpected value for summary: {summary}, expected str one of: {SUMMARY_TYPES}")
    if summary in ("sum", "count") and label_col is None:
        raise AttributeError(f"Must provide name label_col for summary type: {summary}")
    if (backend := backend.casefold()) not in PLOT_BACKENDS:
        raise AttributeError(f"Backend not recognized, expected one of: {PLOT_BACKENDS}, but got: {backend}")

    def _plot_facet_grid(graph):
        """Helper function to reduce duplicate code throughout. Plots the summarized data on the various facet grids

        Args:
            graph: seaborn FacetGrid object with data already provided

        Returns:
            fig and axes of the facet grid with the data plotted appropriately
        """
        if summary == "count":  # Plot positive and negative lines
            graph.map_dataframe(
                sns.lineplot,  x=DATE_LABEL, y=POS_LABEL, label=POS_LABEL, color="green", alpha=0.5, legend="auto",
            )
            graph.map_dataframe(sns.lineplot, x=DATE_LABEL, y="negative", label="negative", color="red", alpha=0.5,)
            for _axis in graph.axes.flat:  # Add legend for positive and negative
                _axis.legend(labels=[POS_LABEL, NEG_LABEL], loc="upper left")
        else:  # Plot other summary line
            graph.map_dataframe(sns.lineplot, x=DATE_LABEL, y=summary)
        graph.set_axis_labels("Date", summary.title())
        graph.set_titles(col_template="{col_name}", row_template="{row_name}")
        return graph.fig, graph.axes

    def _plot_overlap_matplotlib(data, hue_label):
        """Helper function to reduce duplicate code by plotting the summarized data lines as overlapped

        Args:
            data: facet_data for the overlapping lines
            hue_label: column that has the "hue" facet

        Returns:
            fig and ax from the plt subplot with the data plotted appropriately
        """
        fig, axis = plt.subplots(1)
        if summary == "count":  # Positive and negative lines for each hue
            sns.lineplot(data=data, x=DATE_LABEL, y=POS_LABEL, hue=hue_label, palette="Greens", ax=axis)
            sns.lineplot(data=data,  x=DATE_LABEL, y=NEG_LABEL, hue=hue_label, palette="Reds", ax=axis)
        else:
            axis = sns.lineplot(data=data, x=DATE_LABEL, y=summary, hue=hue_label)
        axis.set(xlabel="Date", ylabel=summary)
        return fig, axis

    def _plot_overlap_plotly(data, hue_label):
        """Helper function to reduce duplicate code by plotting the summarized data lines as overlapped

        Args:
            data: facet_data for the overlapping lines
            hue_label: column that has the "hue" facet

        Returns:
            fig and ax from the plt subplot with the data plotted appropriately
        """
        if summary == "count":  # Positive and negative lines for each hue
            data = pd.melt(
                data,
                id_vars=[DATE_LABEL, hue_label],
                value_vars=[POS_LABEL, NEG_LABEL],
                value_name="count",
                var_name="label"
            )
            fig = px.line(
                data, x=DATE_LABEL, y="count", line_group=hue_label, color="label", color_discrete_map=COUNT_COLORS
            )
        else:
            fig = px.line(data, x=DATE_LABEL, y=summary, color=hue_label)
        fig.update_layout(xaxis_title="Date", yaxis_title=summary)
        return fig

    def _get_count_kwargs(curr_kwargs):
        curr_kwargs["data_frame"] = pd.melt(
            curr_kwargs["data_frame"],
            id_vars=[i for i in (DATE_LABEL, KEYWORD_LABEL, SOURCE_LABEL) if i in curr_kwargs["data_frame"].columns],
            value_vars=[POS_LABEL, NEG_LABEL],
            value_name=summary,
            var_name="label",
        )
        curr_kwargs["color"] = "label"
        curr_kwargs["color_discrete_map"] = COUNT_COLORS
        return curr_kwargs

    # Processes the four possible options of faceting, both, one, other, or none
    # Creates plots that separates the data across the facets by drawing individual lines for each group of data
    # If overlap is True, all lines will be drawn on the same plot
    # If overlap is False, the plots will be faceted using seaborn FacetGrid corresponding to the facet options and
    # one line will be drawn on each plot
    if backend == "matplotlib":
        if facet_keywords and facet_sources:
            if overlap:
                # Combine facet label to create a (M x N)-unique list of names for facetting on a single column
                facet_data["Keyword Source"] = facet_data[KEYWORD_LABEL] + "_" + facet_data[SOURCE_LABEL]
                fig, axis = _plot_overlap_matplotlib(facet_data, "Keyword Source")
            else:
                graph = sns.FacetGrid(
                    data=facet_data, row=KEYWORD_LABEL, col=SOURCE_LABEL, hue=KEYWORD_LABEL, sharey=False, aspect=2
                )
                fig, axis = _plot_facet_grid(graph)
        elif facet_keywords:
            if overlap:
                fig, axis = _plot_overlap_matplotlib(facet_data, KEYWORD_LABEL)
                axis.legend(title="Keywords")
            else:
                graph = sns.FacetGrid(data=facet_data, row=KEYWORD_LABEL, hue=KEYWORD_LABEL, sharey=False, aspect=2)
                fig, axis = _plot_facet_grid(graph)
        elif facet_sources:
            if overlap:
                fig, axis = _plot_overlap_matplotlib(facet_data, SOURCE_LABEL)
                axis.legend(title="Sources")
            else:
                graph = sns.FacetGrid(data=facet_data, row=SOURCE_LABEL, hue=SOURCE_LABEL, sharey=False, aspect=2)
                fig, axis = _plot_facet_grid(graph)
        else:
            # If no facets are used we can have some fun by customizing more since everything will be drawn on a single
            # graph that is more or less well-defined compared to the others
            fig, axis = plt.subplots(1)
            if summary == "count":
                sns.lineplot(data=facet_data, x=DATE_LABEL, y=POS_LABEL, color="green", ax=axis)
                sns.lineplot(data=facet_data, x=DATE_LABEL, y=NEG_LABEL, color="red", alpha=0.5, ax=axis)
                axis.legend(labels=[POS_LABEL, NEG_LABEL], loc="upper left")
            else:
                sns.lineplot(data=facet_data, x=DATE_LABEL, y=summary, ax=axis)
            axis.set_xlabel("Date")
            axis.set_ylabel(summary)
            axis.set_title(
                f"{summary} on {facet_data.shape[0]} texts between {facet_data[date_col].min().date()} and"
                f" {facet_data[date_col].max().date()}".title()
            )
    elif backend == "plotly":
        plot_kwargs = {
            "data_frame": facet_data,
            "x": DATE_LABEL,
            "y": summary,
            "labels": {DATE_LABEL: "Date"},
        }
        if facet_keywords and facet_sources:
            if overlap:
                # Combine facet label to create a (M x N)-unique list of names for facetting on a single column
                facet_data["Keyword Source"] = facet_data[KEYWORD_LABEL] + "_" + facet_data[SOURCE_LABEL]
                fig = _plot_overlap_plotly(facet_data, "Keyword Source")
            else:
                if summary == "count":
                    plot_kwargs = _get_count_kwargs(plot_kwargs)
                else:
                    plot_kwargs["color"] = KEYWORD_LABEL
                fig = px.line(**plot_kwargs, facet_row=KEYWORD_LABEL, facet_col=SOURCE_LABEL)
                fig.update_layout(legend_title_text="Keywords")
        elif facet_keywords:
            if overlap:
                fig = _plot_overlap_plotly(facet_data, KEYWORD_LABEL)
            else:
                if summary == "count":
                    plot_kwargs = _get_count_kwargs(plot_kwargs)
                else:
                    plot_kwargs["color"] = KEYWORD_LABEL
                fig = px.line(**plot_kwargs, facet_row=KEYWORD_LABEL)
            fig.update_layout(legend_title_text="Keywords")
        elif facet_sources:
            if overlap:
                fig = _plot_overlap_plotly(facet_data, SOURCE_LABEL)
            else:
                if summary == "count":
                    plot_kwargs = _get_count_kwargs(plot_kwargs)
                else:
                    plot_kwargs["color"] = SOURCE_LABEL
                fig = px.line(**plot_kwargs, facet_row=SOURCE_LABEL)
            fig.update_layout(legend_title_text="Sources")
        else:
            # If no facets are used we can have some fun by customizing more since everything will be drawn on a single
            # graph that is more or less well-defined compared to the others
            if summary == "count":
                plot_kwargs = _get_count_kwargs(plot_kwargs)
            fig = px.line(**plot_kwargs)
            fig.update_layout(
                title=summary.title(),
                xaxis_title="Date",
                yaxis_title=summary,
            )
        if facet_keywords or facet_sources:
            fig.update_yaxes(matches=None)
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    else:
        raise ValueError(f"Plotting backend {backend} not recognized!")
    return fig
