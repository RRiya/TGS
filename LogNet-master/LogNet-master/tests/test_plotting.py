import pytest

import numpy as np
import matplotlib
matplotlib.use('Agg')
from lognet.utilities.Plotting import plot_curves

def test_plot_curves():

    depth = np.linspace(0, 1, 101)
    curve = 5.0 * depth
    curve_title = 'Test'
    color = 'blue'
    tick_labels = True

    figure = matplotlib.pyplot.figure(figsize=(13,12))

    ax = figure.add_subplot(1, 2, 1)

    ax = plot_curves(ax, depth, curve, curve_title=curve_title, color=color, tick_labels=tick_labels)

    pass


def test_plot_formations():
    pass 

