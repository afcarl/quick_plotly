import plotly.offline as ol
import plotly.graph_objs as go

from plotly.tools import get_subplots   
from plotly.tools import FigureFactory as FF

import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde
 

def initialize_plotly():
    ol.init_notebook_mode() 

def default_layout_kwargs():
    """
    Default layout arguments
    """
    return dict(width=1000,
                height=800,
                title='',
                font=dict(color='black', family="\"Open Sans\", verdana, arial, sans-serif", size=16),
                paper_bgcolor='white',
                plot_bgcolor='white',
                hovermode='closest',
                bargap=.05,
                showlegend=True)

def default_geo_kwargs():
    return dict(projection=dict(type='albers usa'), 
                showframe=True, 
                showlakes=True, 
                coastlinecolor='black',
                countrywidth=1,
                countrycolor='rgb(128,128,128)',
                showsubunits=True,
                bgcolor='white',
                showrivers=False,
                subunitcolor=None,
                showcountries=True,
                riverwidth=2, 
                scope='usa',
                rivercolor='rgb(0,0,200)',
                subunitwidth=1,
                showocean=True,
                oceancolor='rgb(220,220,256)',
                lakecolor='white',
                showland=True, 
                framecolor='rgb(128,128,128)',
                coastlinewidth=.5,
                landcolor='rgb(128,128,128)',
                showcoastlines=True,
                framewidth=.5,
                resolution=100)

def update_kwargs(dict_orig, dict_update):
    for k, v in dict_update.items():
        if k in dict_orig:
            dict_orig[k] = v
    return dict_orig


def show_geo_projections():
    """
    Display the projection options provided to plotly.graph_objs.Layout['geo']['projection']['type']
    """
    print """
    'equirectangular'
    'mercator'
    'orthographic'
    'natural earth'
    'kavrayskiy7'
    'miller' 
    'robinson' 
    'eckert4' 
    'azimuthal equal area'
    'azimuthal equidistant'
    'conic equal area'
    'conic conformal'
    'conic equidistant'
    'gnomonic'
    'stereographic'
    'mollweide'
    'hammer'
    'transverse mercator'
    'albers usa'
    """

def kde(x, x_grid, bandwidth=0.4, **kwargs):
    """Kernel Density Estimation with Scipy"""
    # From https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde.evaluate(x_grid)



def bubble_chart(df, x=None, y=None, 
                 marker_size=None, marker_color=None, marker_text=None,
                 colorscale='Viridis', title='', 
                 figsize=(800,800), 
                 outfile=None):
    """
    Plots two dimensions of data in <df>.
    
    Parameters
    ----------
    df: pandas dataframe. 
        The first two columns will be used for visualization.
        
    x: str
        The name of the column associated with the horizontal axis.
        If not provided, we choose first column of <df>
    
    y: str
        The name of the column associated with the vertical axis.
        If not provided, we choose first column of <df> after x
        is removed.
    
    marker_size: either a) int, b) function(df), c) array-like, d) None
        If a function, we evaluate on <df> to determine marker size. If
        an integer, all locations are plotted in the provided <marker_size>.
        If an array-like, then markers are sized according to the array.
        Otherwise, we default to markers of size 5
        
    marker_color: either a) function(df), c) array-like, d) None
        If a function, we evaluate on <df> to determine marker color.
        If an array-like, then markers are sized according to the array.
        Otherwise, we default to markers color to 'rgba(0,116,217,.5)'
        
    marker_text: can be a) a function(df), b) a str, or c) None
        Used to determine each point's hover string. If a function, then
        it is evaluated on <df> to determine each point's string; must
        return a list of strings, each corresponding to a row in <df>.
        If a str type, then all datapoints are named with the provided
        value. If None, we provide only the (lon,lat) values are used.
        
    marker_name: can be function(df) or None.
       Used to determine the name of each point's name for the legend
       if None, the legend is suppressed.
       
    colorscale: string
        the name of a seaborn / colorbrewer colorscale
        
    outfile: filepath str
        If provided, output to an HTML file at provided location
        
    Example
    -------
    from sklearn.datasets import make_classification
    N_FEATURES = 4
    X, y = make_classification(n_samples=100, n_clusters_per_class=1, n_classes=4, n_features=N_FEATURES)

    df = pd.DataFrame(X, columns=['feature_%d' % f for f in range(N_FEATURES)])
    df['class'] = y
    
    # Plot classes, vary size with respect to feature_2
    p = bubble_chart(df,
                 x='feature_1',
                 y='feature_2',
                 marker_size=lambda x: 20*np.abs(x['feature_2'].values), 
                 marker_color=lambda x: x['class'].values,
                 marker_text=get_text,
                 figsize=(1024,512),
                 title='Bubble Chart')
    """
    
    columns = df.columns.values.tolist()
    if x is None:
        x = columns[0]
        columns.remove(x)
        
    if y is None:
        y = columns[0]
    
    n_samples = df.shape[0]
    # use OpenGL for large datasets
    if n_samples > 500:
        scatter_fun = go.Scattergl
    else:
        scatter_fun = go.Scatter

    if hasattr(marker_size, '__call__'):    # function
        sizes = marker_size(df)
    elif isinstance(marker_size, int):      # integer
        sizes = [marker_size]*n_samples
    elif hasattr(marker_size, '__iter__'):  # list or array
        sizes = marker_size
    else:  # default is 10
        sizes = [10]*n_samples
        
    if hasattr(marker_color, '__call__'):            # function
        colors = marker_color(df)
    elif hasattr(marker_color, '__iter__'):          # list or array
        colors = marker_color
    else:
        colors = ['rgba(0,116,217,.5)']*n_samples  # default is blue
    
    if hasattr(marker_text, '__call__'):  # a function
        texts = marker_text(df)
    elif isinstance(marker_text, str):    # a string
        texts = [marker_text]*n_samples 
    else:                                 # default
        texts = []
        
    x_vals = list(df[x].T)
    y_vals = list(df[y].T)
    
    traces = []
    trace = scatter_fun(x=x_vals, y=y_vals, 
                        mode='markers',
                        text=texts,
                        marker=dict(symbol='circle',
                                    color=colors,
                                    colorscale=colorscale,
                                    size=sizes))
    traces.append(trace)

    layout = go.Layout(title=title,
                       width=figsize[0],
                       height=figsize[1],
                       yaxis=go.YAxis(title=y),
                       xaxis=go.XAxis(title=x))
    
    fig = go.Figure(data=traces, layout=layout)
    ol.iplot(fig, show_link=False)
    
    # write figure to HTML file
    if outfile:
        print('Exporting copy of figure to %s...' % outfile)
        ol.plot(fig, auto_open=False, filename=outfile)


def plot_classes(df, classes=None, x=None, y=None, 
                 cmap='colorblind', marker_size=None, 
                 marker_text=None, title='', figsize=(1024,512),
                 outfile=None):
    """
    Plots first two dimensions of <df>, while grouping according to 
    <classes>. If <classes> is None, we look for "classes" column in
    <df>. If <classes> is a string, we use that column of that name
    as the class variable. <classes> can also be an array of unique
    identifiers.
    
    Parameters
    ----------
    df: pandas dataframe. 
        The first two columns will be used for visualization.
        
    classes: a) None, b) str, c) array of unique identifiers
        each entry corresponds to the rows of <df>; entries can
        be strings or ints type (not mixed).
 
    x: str
        The name of the column associated with the horizontal axis.
        If not provided, we choose first column of <df>
    
    y: str
        The name of the column associated with the vertical axis.
        If not provided, we choose first column of <df> after x
        is removed.
    
 
    cmap: string
        the name of a seaborn / colorbrewer colormap
        
    marker_text: function(df, mask)
        function that is applied to <df> and a mask (determined
        by <classes>) to determine the tooltip string on hover
        
    marker_size: function(df, mask)
        function that is applied to <df> and a mask (determined
        by <classes>) to determine the size of datapoints
        
    Example
    -------
    from sklearn.datasets import make_classification
    import pandas as pd
    N_FEATURES = 4
    X, y = make_classification(n_samples=100, n_clusters_per_class=1, n_classes=4, n_features=N_FEATURES)
    df = pd.DataFrame(X)
    df['classes'] = y
    
    # plot the classes 4 different ways
    classes = ['Cluster %d' % (c + 1) for c in y]
    plot_classes(df, classes, title='With Classes List of strs')
    plot_classes(df, y, title='With Classes List of ints')
    plot_classes(df, title="Implicitly Use 'classes' column")
    plot_classes(df, 'classes', title="Explicitly Use 'classes' column")
    
    """
    
    columns = df.columns.values.tolist()
    if x is None:
        x = columns[0]
        columns.remove(x)
        
    if y is None:
        y = columns[0]
        
    if isinstance(classes, str):
        try:
            classes = df[classes].values
        except Exception as e:
            print 'Could not find %s in dataframe' % classes
    elif hasattr(classes, '__call__'):
        classes = classes(df)
    elif classes is None:
        try:
            classes = df['classes'].values
        except Exception as e:
            print 'Could not find <classes> column in dataframe:\m%s' % e
    else:  # classes an array-like
        pass
    
    unique_classes = np.unique(classes)
    unique_classes.sort()
    num_classes = len(unique_classes)
    
    # use OpenGL for large datasets
    if df.shape[0] > 500:
        scatter_fun = go.Scattergl
    else:
        scatter_fun = go.Scatter
    
    # set up colors
    palette = sns.color_palette(cmap, num_classes + 1)
    # rescale to 0 - 256
    rgb = [[int(p*256) for p in color] for color in palette]
    alpha = .5
    marker_rgba = [r+[alpha] for r in rgb]
    line_rgba = [r+[1.] for r in rgb]
    
    traces = []
    for i, c in enumerate(unique_classes):

        # choose color
        if isinstance(c, int) and c < 0:  # for noise classes
            marker_color = 'rgba(0,0,0,%1.2f)' % alpha
            line_color = 'rgba(0,0,0,1.)' % alpha
        else:
            marker_color ='rgba('+ ','.join([str(v) for v in marker_rgba[i]]) +')'
            line_color ='rgba('+ ','.join([str(v) for v in line_rgba[i]]) +')'
            
        mask = [cc == c for cc in classes]

        # isolate data
        class_data = df.loc[mask,:].values
        
        if marker_text:
            texts = marker_text(df, mask)
        else:
            texts = []
            
        if marker_size:
            sizes = marker_size(df, mask)
        else:
            sizes = 16
        
        if isinstance(c, int):
            class_name = 'Class %d' % c
        else:
            class_name = c
            
        trace = scatter_fun(x=list(class_data.T[0]), 
                            y=list(class_data.T[1]), 
                            mode='markers',
                            text=texts,
                            marker=dict(color=marker_color,
                                        line=dict(color=line_color,
                                                  width=1),
                                       symbol='circle',
                                       size=sizes),
                            name=class_name)
        traces.append(trace)
        
    layout = go.Layout(title=title,
                       width=figsize[0],
                       height=figsize[1],
                       yaxis=go.YAxis(title=y),
                       xaxis=go.XAxis(title=x))
    
    fig = go.Figure(data=traces, layout=layout)
    ol.iplot(fig, show_link=False)
    
    # write figure to HTML file
    if outfile:
        print('Exporting copy of figure to %s...' % outfile)
        ol.plot(fig, auto_open=False, filename=outfile)


from plotly.tools import get_subplots   
from scipy.stats import gaussian_kde
def kde(x, x_grid, bandwidth=0.4, **kwargs):
    """Kernel Density Estimation with Scipy"""
    # From https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde.evaluate(x_grid)


def scatter_matrix(df, marker_color=None, 
                   figsize=(1024,1024), 
                   title='', 
                   outfile=None,
                   cmap='deep', ):
    """
    Plot pairwise relationships between all columns in the dataframe
    <df>
    
    Paramters
    ---------
    df: pandas dataframe
        Data to visualize
    
    marker_color: can be a) a string indicating a column in <df>; b) an 
            array-like of values, each corresponding to a row in <df>,
            or c) None, for which each variable gets its own color
    
    title: str
        The figure title
    
    figsize: tuple
        The (height, width) of the figure
        
    cmap: str
        The name of the colormap used (i.e. colorbrewer)
    
    outfile: filepath str
        If provided, output to an HTML file at provided location
        
    
    Example
    -------
    from sklearn.datasets import make_classification
    N_FEATURES = 4
    X, y = make_classification(n_samples=100, n_clusters_per_class=1, n_classes=4, n_features=N_FEATURES)

    df = pd.DataFrame(X, columns=['feature_%d' % f for f in range(N_FEATURES)])
    df['class'] = y
    
    scatter_matrix(df, colors='class', title='Scatter Matrix')

    """
    
    if isinstance(marker_color, str):
        tmp = marker_color
        marker_color = df[marker_color].values.tolist()
        del df[tmp]
    
    columns = df.columns
    n_columns = len(columns)
    
    alpha = .5
    
    color_is_rgba = False
    if hasattr(marker_color, '__call__'): # function provided
        colors = marker_color(df)
        
    elif hasattr(marker_color, '__iter__'):
        if isinstance(marker_color[0], str): # array-like of strings (i.e. categories) 
            unique_vals = np.unique(marker_color).tolist()
            n_colors = len(unique_vals)
            
            # set up colors
            palette = sns.color_palette(cmap, n_colors)
            rgb = [[int(p*256) for p in color] for color in palette]

            colors = []
            
            for c in marker_color:
                colors.append(rgb[unique_vals.index(c)])
                
            colors = ['rgba(' + ','.join([str(c) for c in color])+',%1.2f)'%alpha for color in colors]
        else:
            colors = marker_color
        
    if marker_color is None:

        # set up colors
        palette = sns.color_palette(cmap, n_columns + 1)

        # rescale to 0 - 256
        rgb = [[int(p*256) for p in color] for color in palette]
        marker_rgba = [r+[alpha] for r in rgb]

    
    subplots = range(1,n_columns**2 + 1)
    subplot_idx = 0
    data = []
    # setup subplots
    fig = get_subplots(rows=n_columns, columns=n_columns, 
                       horizontal_spacing=0.05, 
                       vertical_spacing=0.05)

    for i in range(1,n_columns + 1):
        
        if marker_color is None:
            row_color ='rgba('+ ','.join([str(v) for v in marker_rgba[i]]) +')'
            scatter_color = row_color
        else:
            row_color = 'gray'
            scatter_color = colors
            
        
        x_column = df.columns[i-1]
        
        for j in range(1, n_columns + 1):
            y_column = df.columns[j-1]
            if i==j:  # plot histogram and kde along diagonal

                x = df[x_column]
                x_grid = np.linspace(x.min(), x.max(), 100)
                sub_plot = [go.Histogram(x=x, histnorm='probability density', 
                                         marker=go.Marker(color=row_color)), \
                            go.Scatter(x=x_grid, y=kde(x.as_matrix(), x_grid), \
                            line=go.Line(width=2, color='black'))]
            
            else:  # scatter plot
                sub_plot = [go.Scatter(x=df[x_column], y=df[y_column], 
                                    mode='markers',
                                    marker=go.Marker(size=6, 
                                                     color=scatter_color,
                                                     colorscale=cmap))]  # colorscale gets ignore if rgba() provided

            # set text for each datapoint
            for pt in sub_plot:
                pt.update(name='{0}'.format(x_column),\
                          xaxis='x{}'.format(subplots[subplot_idx]),\
                          yaxis='y{}'.format(subplots[subplot_idx]))
                if i!=j:
                    pt.update(text='{0}<br>vs<br>{1}'.format(x_column,y_column))
                
            subplot_idx += 1
            data += sub_plot

    # add x and y labels
    left_index = 1
    bottom_index = 1
    for col in df.columns:
        fig['layout']['xaxis{}'.format(left_index)].update(title=col)
        fig['layout']['yaxis{}'.format(bottom_index)].update(title=col)
        left_index=left_index + 1
        bottom_index=bottom_index + n_columns

    # Remove legend by updating 'layout' key
    fig['layout'].update(showlegend=False, height=figsize[1],width=figsize[0],title=title)
    fig['data'] = go.Data(data)
    ol.iplot(fig, show_link=False)

    # write figure to HTML file
    if outfile:
        print('Exporting copy of figure to %s...' % outfile)
        ol.plot(fig, auto_open=False, filename=outfile)

cols = ['total_followers','total_following','total_published_mixes']
scatter_matrix(np.log1p(beh.data.loc[::skip_every,cols]),
               marker_color=beh.data.loc[::skip_every, 'user_type'].values.tolist(),
               figsize=(800,800))

def box_plot(df, groupby=None, val=None, figsize=(1024,512), title='', ylabel=''):
    """
    Visualize box plots for all columns in the dataframe <df>.
    
    Parameters
    ----------
    df: pandas dataframe
        each column represents a categorical variable; samples are
        along columns.
        
    groupby: str
        The name of a column to define groups. If None provided,
        we assume equal sample sizes and use columns as groups.

    val: str
        Used in conjuction with <groupby> to choose column values to plot.
        If None, then we choose the first column (excluding <groupby>)
    
    figsize: tuple
        The (width, height) of the figure in pixels
    
    title: str
        The figure title
    
    ylabel: str
        The name of the y axis

    outfile: filepath str
        If provided, output to an HTML file at provided location


    Example
    -------
    from sklearn.datasets import make_classification
    N_FEATURES = 4
    X, y = make_classification(n_samples=100, n_clusters_per_class=1, n_classes=4, n_features=N_FEATURES)

    df = pd.DataFrame(X, columns=['feature_%d' % f for f in range(N_FEATURES)])
    df['class'] = y
    
    box_plot(df, title='Box Plot')

    """
    layout = go.Layout(title=title, 
                       height=figsize[1], 
                       width=figsize[0], 
                       yaxis=go.YAxis(title=ylabel))

    data = []
    if groupby is None:
        for col in df.columns:
            data.append(go.Box(y=df[col], name=col))
    else:
        groups = df[groupby].unique().tolist()
        if val is None:
            val = df.columns.drop(groupby)[0] # choose first non-groupby column
        for group in groups:
            mask = df[groupby] == group
            data.append(go.Box(y=df.loc[mask, val], name=group))
            

    ol.iplot(go.Figure(data=data, layout=layout), show_link=False)


def bar_plot(df, x, y, group_by=None, 
             orientation='v', barmode='stack', 
             title='', cmap='Set2', bargap=0.05,
             figsize=(1024,512), agg_function='sum',
             ylabel=None,
             xlabel=None,
             outfile=None):
    """
    Create bar plot with grouping capability
    
    Parameters
    ----------
    df:  pandas dataframe
        Data to visualize

    x: str
        The name of a column in <df> that corresponds 
        to categorical data
    
    y: str
        The name of a column in <df> that corresponds to 
        the bar value
    
    group_by: str
        The name of a column in <df> to perform an additional
        categorical grouping

    agg_function: str 'mean' | 'count' | 'median' | 'sum'
        When no group_by columns provided, we apply the 
        aggregation across remaining columns. 
    
    orientation: str, either 'v' (default) or 'h'
        Specify vertical or horizontal orientation of the
        plot
    
    barmode: str, either 'stack' or 'group'
        Specify grouped or stacked bar graph
    
    title: str
        HTML-formatted string for the figure title

    xlabel: str
        HTML-formatted string for the x-axis title

    ylabel: str
        HTML-formatted string for the y-axis title    

    cmap: str
        The name of a colorbrewer colormap
    
    bargap: float in (0,1)
        The amount of separation of bars in proportion of
        bar width
    
    figsize: tuple
        The (width, height) size of the figure in pixels.

    outfile: filepath str
        If provided, output to an HTML file at provided location
        
    Example
    -------
    import pandas as pd
    
    animals = pd.DataFrame([
              ['cat',10, 'housepet'],
              ['dog',20,'housepet'],
              ['fish',5,'housepet'],
              ['cat',20,'zooanimal'],
              ['dog',50,'zooanimal'],
              ['fish',20,'zooanimal'],], columns=['animal','value','group'])
    
    bar_plot(animals, x='group',y='value', group_by='animal', 
         orientation='v', cmap='Set2', barmode='stack', 
         title='Stacked Bar Plot',
         figsize=(800,512))
    """
    if 'h' in orientation:
        x, y = y, x
        
    if xlabel is None:
        xlabel=x
        
    if ylabel is None:
        ylabel=y
        
    layout = go.Layout(title=title,  
                       yaxis=go.YAxis(title=ylabel),
                       xaxis=go.XAxis(title=xlabel),
                       barmode=barmode,
                       bargap=bargap,
                       height=figsize[1],
                       width=figsize[0])
    if group_by:

        groups = df[group_by].unique()
        n_groups = len(groups)
        # set up colors
        palette = sns.color_palette(cmap, n_groups + 1)
        # rescale to 0 - 256
        rgb = [[int(p*256) for p in color] for color in palette]

        data = []
        for i, group in enumerate(groups):
            color ='rgb('+ ','.join([str(v) for v in rgb[i]]) +')'
            xs = df[df[group_by] == group][x].values
            ys = df[df[group_by] == group][y].values
            data.append(go.Bar(x=xs, y=ys, name=group, 
                               marker=dict(color=color),
                               orientation=orientation),)
    else:
        # there's probably a much better way of doing this using .apply()
        if agg_function == 'sum':
            data = [go.Bar(x=sorted(df[x].unique().sort()), y=df.groupby([x]).sum()[y].values, orientation=orientation)]
        elif agg_function == 'mean':
            data = [go.Bar(x=sorted(df[x].unique()), y=df.groupby([x]).mean()[y].values, orientation=orientation)]
        elif agg_function == 'median':
            data = [go.Bar(x=sorted(df[x].unique()), y=df.groupby([x]).median()[y].values, orientation=orientation)]
        elif agg_function == 'count':
            data = [go.Bar(x=sorted(df[x].unique()), y=df.groupby([x]).count()[y].values, orientation=orientation)]
        elif agg_function == 'min':
            data = [go.Bar(x=sorted(df[x].unique()), y=df.groupby([x]).min()[y].values, orientation=orientation)]
        elif agg_function == 'max':
            data = [go.Bar(x=sorted(df[x].unique()), y=df.groupby([x]).max()[y].values, orientation=orientation)]
        
    fig = go.Figure(data=data, layout=layout)
    ol.iplot(fig, show_link=False)

    # write figure to HTML file
    if outfile:
        print('Exporting copy of figure to %s...' % outfile)
        ol.plot(fig, auto_open=False, filename=outfile)


def pretty_table(df, outfile=None):
    """
    Display pandas dataframe as a nicely-formated HTML

    Parameters
    ----------
    outfile: filepath str
        If provided, output to an HTML file at provided location

    Example
    -------
    import pandas as pd
    
    animals = pd.DataFrame([
              ['cat',10, 'housepet'],
              ['dog',20,'housepet'],
              ['fish',5,'housepet'],
              ['cat',20,'zooanimal'],
              ['dog',50,'zooanimal'],
              ['fish',20,'zooanimal'],], columns=['animal','value','group'])

    pretty_table(animals)

    """
    
    table = FF.create_table(df)
    ol.iplot(table, show_link=False)

    # write figure to HTML file
    if outfile:
        print('Exporting copy of figure to %s...' % outfile)
        ol.plot(table, auto_open=False, filename=outfile)


def bubble_map(df, 
               marker_size=None, 
               marker_color=None, 
               marker_text=None, 
               marker_name=None, 
               title='',
               scope='usa',
               projection_type='albers usa',
               geo_kwargs=None,
               layout_kwargs=None,
               outfile=None,
               cmap='Set2',
               alpha=.7):
    """
    Scatter plot atop a map.
    
    Parameters
    ----------
    df:   A dataframe of points to plot. 
        Must include columns ['lat','lon'], corresponding to the latitude
        and longitude of the points, respectively.
        
    scope: str
        The viewpoint range of the map.
        
    projection_type: str
        The name of the projection type to use. See https://plot.ly/python/dropdowns/ 
        for details
        
    marker_size: either a) int, b) function(df), c) array-like, d) None
        If a function, we evaluate on <df> to determine marker size. If
        an integer, all locations are plotted in the provided <marker_size>.
        If an array-like, then markers are sized according to the array.
        Otherwise, we default to markers of size 5
        
    marker_color: either a) function(df), c) array-like, d) None
        If a function, we evaluate on <df> to determine marker color.
        If an array-like, then markers are sized according to the array.
        Otherwise, we default to markers color to 'rgba(0,116,217,.5)'
        
    marker_text: can be a) a function(df), b) a str, or c) None
        Used to determine each point's hover string. If a function, then
        it is evaluated on <df> to determine each point's string; must
        return a list of strings, each corresponding to a row in <df>.
        If a str type, then all datapoints are named with the provided
        value. If None, we provide only the (lon,lat) values are used.
        
    marker_name: can be function(df) or None.
       Used to determine the name of each point's name for the legend
       if None, the legend is suppressed.
       
    title: HTML-formatted string
        The title of the figure

    outfile: filepath str
        If provided, output to an HTML file at provided location
        
    Example
    -------
    import pandas as pd
    
    cities = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_us_cities.csv')
       
    # helper functions for color, size, text
    def get_size(df):
    scale = 80000.
    size0 = df['pop'].values / scale
    return [np.max([s, 2]) for s in size0]

    def get_text(df):
        city_names = df['name'].values
        pop_size = df['pop'].values
        return ['%s<br>Population: %d' % (v) for v in zip(city_names, pop_size)]

    def get_color(df):
        colors = []
        for i, row in df.iterrows():
            if row['pop'] >= 1000000:
                colors.append("rgba(0,116,217,.7)")
            elif row['pop'] < 1000000 and row['pop'] >= 500000:
                colors.append("rgba(255,65,54, .7)")
            else:
                colors.append("rgba(133,20,75, .7)")
        return colors
        
    bubble_map(cities,
           projection_type='albers usa',
           scope='usa',
           marker_size=get_size, 
           marker_text=get_text, 
           marker_color=get_color,
           title="Top US City Populations")

    
    For other examples, see https://plot.ly/javascript/scatter-plots-on-maps/

    """
    
    n_locations = len(df)
 
    if hasattr(marker_size, '__call__'):    # function
        sizes = marker_size(df)
    elif isinstance(marker_size, int):      # integer
        sizes = [marker_size]*n_locations
    elif hasattr(marker_size, '__iter__'):  # list or array
        sizes = marker_size
    else:                                   # default 
        sizes = [5]*n_locations
        
    if hasattr(marker_color, '__call__'):            # function
        colors = marker_color(df)
    elif hasattr(marker_color, '__iter__'):          # list or array
        colors = marker_color
    else:
        colors = ['rgba(0,116,217,.5)']*n_locations  # default is blue
    
    if hasattr(marker_text, '__call__'):  # a function
        texts = marker_text(df)
    elif isinstance(marker_text, str):    # a string
        texts = [marker_text]*n_locations 
    else:                                 # default
        texts = []
        
    if hasattr(marker_name, '__call__'):  # a function
        names = marker_name(df)
        show_legend=True
        unique_names = sorted(np.unique(names))
        num_classes = len(unique_names)
        # set up colors
        palette = sns.color_palette(cmap, num_classes + 1)
        # rescale to 0 - 256
        rgb = [[int(p*256) for p in color] for color in palette]
        marker_rgba = [r+[alpha] for r in rgb]
        line_rgba = [r+[1.] for r in rgb]
    else:                                 # show nothing
        names = []
        show_legend=False

    locations = []
    
    if len(names) > 0:
        
        for i, un in enumerate(unique_names):
            marker_color ='rgba('+ ','.join([str(v) for v in marker_rgba[i]]) +')'
            line_color ='rgba('+ ','.join([str(v) for v in line_rgba[i]]) +')'

            mask = [n == un for n in names]
            name_data = df[mask]
            name_text = [texts[i] for i, m in enumerate(mask) if m is True]
            name_sizes = [sizes[i] for i, m in enumerate(mask) if m is True]
            
            location = dict(type='scattergeo',
                    locationmode='ISO-3',
                    lon=name_data['lon'],
                    lat=name_data['lat'],
                    text=name_text,
                    name=un,
                    marker=dict( 
                                size=name_sizes, 
                                color=marker_color,
                                line=dict(color=line_color, width=1)
                            )
                    )
        
            locations.append(location)
        
    else:
        location = dict(
            type='scattergeo',
            locationmode='ISO-3',
            lon=df['lon'],
            lat=df['lat'],
            text=texts,
            name=names,
            marker=dict( 
                        size=sizes, 
                        color=colors,
                        colorscale=cmap)
                    )

        locations.append(location)

    geo_args = default_geo_kwargs()
    if geo_kwargs:
        geo_args = update_kwargs(geo_args, geo_kwargs)

    layout_args = default_layout_kwargs()
    if layout_kwargs:
        layout_args = update_kwargs(layout_args, layout_kwargs)

    layout_args['title'] = title
    layout_args['showlegend'] = show_legend
    layout_args['geo'] = dict(**geo_args) 
    layout = dict(**layout_args)


    fig = dict(data=locations, layout=layout)
    ol.iplot(fig, show_link=False)

    # write figure to HTML file
    if outfile:
        print('Exporting copy of figure to %s...' % outfile)

def line_chart(df, x, y=None, 
               line_colors=None, 
               line_names=None, 
               ylabel='', xlabel='',
               title='', cmap='deep', 
               outfile=None):
    """
    Plot 2D traces of values along the columns of the dataframe <df>
    
    Parameters
    ----------
    df: pandas dataframe with at least two columns
        The data to plot
        
    x: str
        Name of the domain (horizontal-axis) variable
    
    y: str, list of strings, or None (default)
        The name columns to use as the range (vertical-axis) variables.
        If None provided, we use the remaining columns in <df>
        
    line_colors: function, array-like of 'rgb()' or numerical values, or None
        Determines the colors of the traces. Can explicitly provide a list
        of rgb values or numerical values. If None, we construct trace colors
        from <cmap>
        
    cmap: str
        The name of a colorbrewer color scheme
        
    line_names: list or None
        Alternative names to the traces.  If None, we use the names of the
        <y> columns as names.
        
    ylabel: HTML-formatted str
        The name of the vertical axis
        
    xlabel: HTML-formatted str
        The name of the horizontal axis
        
    title: HTML-formatted str
        The name of the figure

    outfile: filepath str
        If provided, output to an HTML file at provided location
        
    Example
    -------
    import pandas as pd
    import numpy as np
    
    # set up some fake traces
    xx = np.linspace(-np.pi*2,np.pi*2, 100)
    cos_y = np.cos(xx)
    cos2_y = np.cos(xx**2)
    sin_y = np.sin(xx)
    sin2_y = np.sin(xx**2)

    functions = pd.DataFrame(np.vstack([xx, cos_y, cos2_y, sin_y, sin2_y])).T
    functions.columns = ['x','cos(x)','cos(x^2)', 'sin(x)', 'sin(x^2)']
    line_chart(functions, x='x', title='Line Chart', xlabel='x', ylabel='f(x)')

    
    """
    
    x_val = df[x].values
    
    if isinstance(y, str):  # single column as string
        y = [y]
    elif y is None:  # use remaining columns as traces
        y = df.drop([x], axis=1).columns
        
    n_columns = len(y)
    
    if line_names is None:
        line_names = y
    
    # set up line colors
    if hasattr(line_colors, '__call__'):    # function
        line_colors = line_colors(df)
    elif hasattr(line_colors, '__iter__'):  # list or array
        pass
    elif line_colors is None:  # set up color palette
         # set up colors
        palette = sns.color_palette(cmap, n_columns)

        # rescale to 0 - 256
        rgb = [[int(p*256) for p in color] for color in palette]
        # define line colors
        line_colors = ['rgb(%d,%d,%d)' % (v[0],v[1],v[2]) for v in rgb]
    
    traces = []
    for i, col in enumerate(y):
        y_val = df[col].values
        line_color = line_colors[i]
        trace = go.Scatter(x=x_val,
                           y=y_val,
                           name=line_names[i],
                           line=(dict(color=line_color, shape='spline')))
        
        traces.append(trace)
    
    layout = dict(title=title,
                  xaxis=dict(title=xlabel),
                  yaxis=dict(title=ylabel))
    
    fig = dict(data=traces, layout=layout)
    ol.iplot(fig, show_link=False)

    # write figure to HTML file
    if outfile:
        print('Exporting copy of figure to %s...' % outfile)
        ol.plot(fig, auto_open=False, filename=outfile)


def choropleth(df, region_codename,
               region_value=None, 
               region_text=None,
               locationmode='USA-states',
               scope='usa',
               projection_type='albers usa',
               colorscale='Viridis',
               layout_kwargs=None,
               geo_kwargs=None,
               title='',
               colorbar_title=None,
               outfile=None):
    
    """
    Create a chorpleth map visualization of values in dataframe <df>
    
    Parameters
    ----------
    df: pandas dataframe object
        Each row corresponds to a location. Must contain a column of 
        location codes that correspond with the parameter <locationmode>
        
    region_codename: str
        The column in <df> containing the region codes
        
    region_value: str, array-like of numbers, function(df), or None
        How we color-code each location. If a string is provided, we 
        use the column in <df> with the corresponding name. If an array
        of numbers, it must have the same length as <df>. If a function,
        it must take in <df> and return an array the same length as <df>
        If None, we default to all regions being colored according to 
        region_value == 1
            
    location_mode: str: "ISO-3" | "USA-states" (default) | "country names"
        The mode used to identify values in <region_codename> to regions
        in the map.
        
    scope: str: "world" | "usa" (default) | "europe" | "asia" | "africa" | 
             "north america" | "south america"
        Isolated region in the world map to focus visualization.
    
    projection_type: str
        The type of map projection used. Defualt is 'albers usa'. Run 
        plotly_api.show_geo_projections() to display full list of options
    
    region_text: str, array-like of strings, function(df), or None
        How we determine tool-tipe for each location. If a string is 
        provided, we use the column in <df> with the corresponding name. 
        If an array of strings, it must have the same length as <df>. 
        If a function, it must take in <df> and return an array the same 
        length as <df> If None, we default to no text.
        
    colorscale: str
        The name of a colorbrewer colorscale used to color the map. Default
        is 'Viridis'
        
    geo_kwargs: dict
        Optional geography keyword arguments. These will be used to update
        the default set of arguments. For details see 
        https://plot.ly/python/reference/#layout
        
    geo_kwargs: dict
        Optional Layout keyword arguments. These will be used to update
        the default set of arguments. For details see
        https://plot.ly/python/reference/#layout
        
    title: HTML-formatted string
        Figure title
        
    colorbar_title: str or None
        If None, we try determine the title from <region_value>.

    outfile: filepath str
        If provided, output to an HTML file at provided location
        
    Example
    -------
    import pandas as pd
    exports = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv')
    
    val = 'total exports'
    
    # California wins
    choropleth(exports, 
               region_codename='code', 
               region_value=val, 
               region_text='state',
               title='US Total Exports by State')  
               
    """
    
    n_locations = len(df)
    
    # parse geo kwargs
    geo_args = default_geo_kwargs()
    if geo_kwargs:
        geo_args = update_kwargs(geo_args, geo_kwargs)
    geo_args['scope'] = scope
    geo_args['projection'] = dict(type=projection_type)
    
    # parse layout kwargs
    layout_args = default_layout_kwargs()
    if layout_kwargs:
        layout_args = update_kwargs(layout_args, layout_kwargs)
      
    # construct Layout, including geography
    layout = dict(**layout_args)
    layout['title'] = title
    layout['geo'] = dict(**geo_args)
    
    if hasattr(region_value, '__call__'):
        values = region_value(df)
    elif hasattr(region_value, '__iter__'):
        values = region_value
    elif isinstance(region_value, str):
        values = df[region_value].values
    elif isinstance(region_value, int):
        values = region_value * np.ones(n_locations)
    else:
        values = np.ones(n_locations)
    
    if hasattr(region_text, '__call__'):
        text = region_text(df)
    elif hasattr(region_text, '__iter__'):
        text = [(str(rt)).strip() for rt in region_text]
    elif isinstance(region_text, str):
        text = [str(t).strip() for t in df[region_text].values]
    else:
        text = []
    
    if colorbar_title:
        colorbar = dict(title=colorbar_title)
    elif isinstance(region_value, str):
        colorbar = dict(title=region_value)
    else:
        colorbar={}
        
    locations = df[region_codename]
    
    data = [dict(type='choropleth',
                 colorscale=colorscale,
                 autocolorscale=False,
                 locations=locations,
                 z=values,
                 text=text,
                 locationmode=locationmode,
                 marker=dict(line=dict(color='rgb(255,255,255)', width=.5)),
                 colorbar=colorbar)]
    
    fig = dict(data=data, layout=layout)
    ol.iplot(fig, show_link=False) 
    
    # write figure to HTML file
    if outfile:
        print('Exporting copy of figure to %s...' % outfile)
        ol.plot(fig, auto_open=False, filename=outfile)


def dist_plot(df, 
              groupby=None,
              val=None,
              bin_size=1, 
              title=None, 
              show_kde=True, show_rug=True, 
              show_legend=True, 
              figsize=None,
              outfile=None):
    
    if groupby is None:
        fig = FF.create_distplot([df[c] for c in df.cols], 
                                 cols, 
                                 bin_size=bin_size,
                                 show_rug=show_rug,
                                 show_curve=show_kde)
    else:
        groups = df[groupby].unique().tolist()
        data = []
        if val is None:
            val = df.columns.drop(groupby)[0] # choose first non-groupby column
            
        for group in groups:
            mask = df[groupby] == group
            data.append(df.loc[mask, val])
            
        fig = FF.create_distplot(data, 
                                 groups, 
                                 bin_size=bin_size,
                                 show_rug=show_rug,
                                 show_curve=show_kde)
        
    fig['layout'].update(showlegend=show_legend)
    
    if title:
        fig['layout'].update(title=title)
        
    if figsize and len(figsize) == 2:
        fig['layout'].update(width=figsize[0])
        fig['layout'].update(height=figsize[1])
        
    ol.iplot(fig, show_link=False)
    
    # write figure to HTML file
    if outfile:
        print('Exporting copy of figure to %s...' % outfile)
        ol.plot(fig, auto_open=False, filename=outfile)
