def import_modules():
    from ipywidgets import interact, FloatSlider, IntSlider, Button, HBox
    import numpy as np
    from numpy import fft
    from scipy import special
    from scipy.interpolate import CubicSpline
    from bokeh.io import push_notebook
    from bokeh.plotting import figure, show, output_notebook
    from bokeh.models import ColumnDataSource, Range1d, LinearAxis, Title, Label, Span
    from bokeh.layouts import column, row
    from bokeh.models.glyphs import Line
    # BOKEH_RESOURCES=inline
    # from bokeh.resources import INLINE
    output_notebook()
    import warnings
    warnings.filterwarnings('ignore')
    return

