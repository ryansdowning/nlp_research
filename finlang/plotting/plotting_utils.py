import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from functools import reduce
from typing import Optional, Dict, Set, Union, List

SOURCE_LABEL = '_SOURCE_LABEL'
KEYWORD_LABEL = '_KEYWORD_LABEL'
SUMMARY_LABEL = '_SUMMARY'
DATE_LABEL = '_DATE'
SUMMARY_TYPES = ('mentions', 'count', 'sum')


def plot_keywords(
        data: pd.DataFrame,
        keywords_linking: Dict[str, Set],
        date_col: str,
        source_col: str,
        label_col: Optional[str] = None,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
        sources: Optional[Union[str, List[str]]] = None,
        keywords: Optional[Union[str, List[str]]] = None,
        frequency: Optional[str] = 'd',
        summary: Optional[str] = 'mentions',
        facet_keywords: Optional[bool] = False,
        facet_sources: Optional[bool] = False,
        overlap: Optional[bool] = False,
):
    """Dynamic plotting function that works with upstream text data that has been tagged in some (numerical) way

    The data provided also has a corresponding keywords_linking that maps keywords to the respective indexes of the data
    table. This helps save a lot of memory with large datasets and also speeds up computation when working with many
    keywords

    The provided functionality allows the user to pass in a dataframe including a date_col, source_col, and label_col,
    and then specify various options for producing summarized graphs that represent the given data. Multiple summarizing
    methods are currently supported and the faceting optionality allows the user to visualize large data in a meaningful
    and clear manner.

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
        frequency: How often to compute summary statistic (must be compatible with pandas groupby) default 'd', daily.
        summary: Summary method, currently supports:
                 mentions - Size of each data group
                 sum - Sum of label_col for each data group
                 count - count of positive and negative values in label_col for each data group
        facet_keywords: Boolean flag, whether or not to divide the plotted lines among the individual keywords
                        Default False, does not facet on keywords. If True, must provide <keywords> parameter
        facet_sources: Boolean flag, whether or not to divide the plotted lines among the individual sources
                       Default False, does not facet on sources. If True, must provide <sources> parameter
        overlap: If the data itself is facetted among the keywords of sources, this flag allows the lines to all be
                 drawn on a single plot rather than using seaborn FacetGrid to draw one line on each subplot
                 Default False, facets plots according to facet options

    Returns:
        fig and ax (or axes) with the data plotted corresponding to the given options
    """
    # Valid Input Checking
    summary = summary.casefold()
    if (summary := summary.casefold()) not in SUMMARY_TYPES:
        raise AttributeError(
            f"Got unexpected value for summary: {summary}, expected str one of: {SUMMARY_TYPES}"
        )
    if summary in ('sum', 'count') and label_col is None:
        raise AttributeError(f"Must provide name label_col for summary type: {summary}")
    if facet_keywords and keywords is None:
        raise AttributeError(
            "If facet_keywords is set to True, you must provide an Iterable of <keywords>s to facet on"
        )
    if facet_sources and sources is None:
        raise AttributeError("If facet_sources is set to True, you must provide an Iterable of <source>s to facet on")

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

    def _compute_summary_data(keyword=None, source=None):
        # Filter data based on keyword and source provided (or not provided)
        if keyword is not None and source is not None:
            idxs = keywords_linking[keyword].intersection(plot_idxs)
            ungrouped = plot_data.loc[idxs]
            ungrouped = ungrouped[ungrouped[source_col] == source]
        elif keyword is not None:
            idxs = keywords_linking[keyword].intersection(plot_idxs)
            ungrouped = plot_data.loc[idxs]
        elif source is not None:
            ungrouped = plot_data[plot_data[source_col] == source]
        else:
            ungrouped = plot_data
        grouped_data = ungrouped.groupby(pd.Grouper(key=date_col, freq=frequency))  # Group by date_col with frequency

        # Create DataFrame with summary data stored in preset columns, count is different because it has 2 summary cols
        if summary == 'mentions':
            mentions_data = grouped_data.size()
            summary_data = pd.DataFrame({DATE_LABEL: mentions_data.index, SUMMARY_LABEL: mentions_data.tolist()})
        elif summary == 'sum':
            sum_data = grouped_data[label_col].sum()
            summary_data = pd.DataFrame({DATE_LABEL: sum_data.index, SUMMARY_LABEL: sum_data.tolist()})
        elif summary == 'count':
            count_data = grouped_data[label_col].apply(lambda x: ((x > 0).sum(), (x < 0).sum()))
            summary_data = pd.DataFrame(count_data.tolist(), columns=('positive', 'negative'), index=count_data.index)
            summary_data[DATE_LABEL] = summary_data.index
            summary_data.reset_index(inplace=True)
        else:
            raise AttributeError(f"Invalid summary value encountered, {summary}")

        # Add column for keyword and source if provided for faceting
        if facet_keywords and keyword is not None:
            summary_data[KEYWORD_LABEL] = keyword
        if facet_sources and source is not None:
            summary_data[SOURCE_LABEL] = source
        return summary_data

    def _plot_facet_grid(g):
        """Helper function to reduce duplicate code throughout. Plots the summarized data on the various facet grids

        Args:
            g: seaborn FacetGrid object with data already provided

        Returns:
            fig and axes of the facet grid with the data plotted appropriately
        """
        if summary == 'count':  # Plot positive and negative lines
            g.map_dataframe(sns.lineplot, x=DATE_LABEL, y='positive', label='posititve', color='green', alpha=.5,
                            legend='auto')
            g.map_dataframe(sns.lineplot, x=DATE_LABEL, y='negative', label='negative', color='red', alpha=.5)
            for ax in g.axes.flat:  # Add legend for positive and negative
                ax.legend(labels=['Positive', 'Negative'], loc='upper left')
        else:  # Plot other summary line
            g.map_dataframe(sns.lineplot, x=DATE_LABEL, y=SUMMARY_LABEL)
        g.set_axis_labels('Date', summary.title())
        g.set_titles(col_template="{col_name}", row_template="{row_name}")
        return g.fig, g.axes

    def _plot_overlap(data, hue_label):
        """Helper function to reduce duplicate code by plotting the summarized data lines as overlapped

        Args:
            data: facet_data for the overlapping lines
            hue_label: column that has the "hue" facet

        Returns:
            fig and ax from the plt subplot with the data plotted appropriately
        """
        fig, ax = plt.subplots(1)
        if summary == 'count':  # Positive and negative lines for each hue
            sns.lineplot(data=data, x=DATE_LABEL, y='positive', hue=hue_label, palette='Greens', ax=ax)
            sns.lineplot(data=data, x=DATE_LABEL, y='negative', hue=hue_label, palette='Reds', ax=ax)
        else:
            ax = sns.lineplot(data=data, x=DATE_LABEL, y=SUMMARY_LABEL, hue=hue_label)
        ax.set(xlabel='Date', ylabel=summary)
        return fig, ax

    # Processes the four possible options of faceting, both, one, other, or none
    # Creates plots that separates the data across the facets by drawing individual lines for each group of data
    # If overlap is True, all lines will be drawn on the same plot
    # If overlap is False, the plots will be faceted using seaborn FacetGrid corresponding to the facet options and
    # one line will be drawn on each plot
    if facet_keywords and facet_sources:
        facet_data = pd.concat([_compute_summary_data(keyword, source) for source in sources for keyword in keywords])
        if overlap:
            # Combine facet label to create a (M x N)-unique list of names for facetting on a single column
            facet_data['Keyword Source'] = facet_data[KEYWORD_LABEL] + "_" + facet_data[SOURCE_LABEL]
            fig, ax = _plot_overlap(facet_data, 'Keyword Source')
        else:
            g = sns.FacetGrid(data=facet_data, row=KEYWORD_LABEL, col=SOURCE_LABEL, hue=KEYWORD_LABEL, sharey=False, aspect=2)
            fig, ax = _plot_facet_grid(g)
    elif facet_keywords:
        facet_data = pd.concat([_compute_summary_data(keyword=keyword) for keyword in keywords])
        if overlap:
            fig, ax = _plot_overlap(facet_data, KEYWORD_LABEL)
            ax.legend(title='Keywords')
        else:
            g = sns.FacetGrid(data=facet_data, row=KEYWORD_LABEL, hue=KEYWORD_LABEL, sharey=False, aspect=2)
            fig, ax = _plot_facet_grid(g)
    elif facet_sources:
        facet_data = pd.concat([_compute_summary_data(source=source) for source in sources])
        if overlap:
            fig, ax = _plot_overlap(facet_data, SOURCE_LABEL)
            ax.legend(title='Sources')
        else:
            g = sns.FacetGrid(data=facet_data, row=SOURCE_LABEL, hue=SOURCE_LABEL, sharey=False, aspect=2)
            fig, ax = _plot_facet_grid(g)
    else:
        # If no facets are used we can have some fun by customizing more since everything will be drawn on a single
        # graph that is more or less well-defined compared to the others
        facet_data = _compute_summary_data()
        fig, ax = plt.subplots(1)
        if summary == 'count':
            sns.lineplot(data=facet_data, x=DATE_LABEL, y='positive', color='green', ax=ax)
            sns.lineplot(data=facet_data, x=DATE_LABEL, y='negative', color='red', alpha=.5, ax=ax)
            ax.legend(labels=['Positive', 'Negative'], loc='upper left')
        else:
            sns.lineplot(data=facet_data, x=DATE_LABEL, y=SUMMARY_LABEL, ax=ax)
        ax.set_xlabel('Date')
        ax.set_ylabel(summary)
        ax.set_title(
            f'{summary} on {plot_data.shape[0]} texts between {plot_data[date_col].min().date()} and'
            f' {plot_data[date_col].max().date()}'.title()
        )
    return fig, ax
