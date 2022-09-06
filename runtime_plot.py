import matplotlib.pyplot as plt
import numpy as np
import seaborn
import pandas as pd


def boxplots_statistics(data1, data2):
    """Print the 5-data summary of running time by Local Search 1 and Local Search 2 respectively
    Parameters
    ----------
    data1: array
            The list of running times by one of the local search approch on a certain graph
    data2: array
            The list of running times by one of the local search approch on a certain graph
    """
    dataset = pd.DataFrame(data={"Summary of Local Search 1 Running Time": data1, "Summary of Local Search 2 Running Time": data2})
    print(dataset.describe())


def boxplots(data1, data2, figure_size=(8, 6), resolution=150):
    """Produce the boxplots of running time by Local Search Method 1 and Local Search Method 2 respectively
        Parameters
        ----------
    data1: array
                The list of running times by one of the local search approch on a certain graph
        data2: array
                The list of running times by one of the local search approch on a certain graph
        figure_size: tuple
                The size of the output plot (e.g. 8 inches x 6 inches)
        resolution: int
                The resolution of the output figure (e.g. 150 dpi (dots per inches))
    """
    # specify the resolution and size of the figure
    plt.figure(num=None, figsize=figure_size, dpi=resolution)

    title = "Boxplots of Running Time for Two Local Searches"
    x_label = "Types of Local Searches"
    y_label = "Running Time (s)"
    x_ticks = ["Local Search 1", "Local Search 2"]

    # produce the box plots
    box_ax = seaborn.boxplot(data=[data1, data2])
    box_ax.set_title(title)
    box_ax.set_xlabel(x_label)
    box_ax.set_ylabel(y_label)
    box_ax.set_xticklabels(x_ticks)

    # save the figure
    plt.show()
    box_ax.figure.savefig("boxplots.eps", format="eps", dpi=resolution)
