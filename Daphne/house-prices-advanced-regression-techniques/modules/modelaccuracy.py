import numpy as np
import os
import pandas as pd
import category_encoders as ce
from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

def add_identity(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes

def allyouneedtoknow(pred, true):
    from sklearn.metrics import r2_score
    R2_LLars = r2_score(pred, true)
    print("Rscore :" + str(R2_LLars))
    from sklearn.metrics import explained_variance_score
    EV_LLars = explained_variance_score(true, pred)
    print("Variance explained: " + str(EV_LLars))

    f, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim((0, max([max(pred), max(true)])))
    ax.set_ylim((0, max([max(pred), max(true)])))
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.scatter(true, pred, c=".3")
    add_identity(ax, color='r', ls='--')
    plt.show()

