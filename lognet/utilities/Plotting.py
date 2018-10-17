import numpy as np
import matplotlib.pyplot as plt

PLOT_CURVES = ['Gamma', 'Resistivity', 'Neutron', 'Sonic', 'Caliper', 'Formation']

FORMATION_PATTERNS = {
    17: {"name": 'Bell Canyon', "color": 'red', "hatch": '/'},
    91: {"name": 'Mississippian', "color": 'green', "hatch": '//'},
    23: {"name": 'Woodford Shale', "color": 'blue', "hatch": '*'},
    15: {"name": 'Rustler', "color": 'cyan', "hatch": '-'},
    16: {"name": 'Salado', "color": 'yellow', "hatch": '.'},
    21: {"name": 'Upper Barnett Shale', "color": 'magenta', "hatch": '\\'},
    24: {"name": 'Fusselman', "color": 'salmon', "hatch": ''},
    19: {"name": 'Bone Spring', "color": 'navajowhite', "hatch": '+'},
    18: {"name": 'Brushy Canyon', "color": 'moccasin', "hatch": 'o-'},
    92: {"name": 'Wolfcamp Shale', "color": 'mediumturquoise', "hatch": '0'},
    25: {"name": 'Simpson', "color": 'deepskyblue', "hatch": '+*'},
    62: {"name": 'Ellenburger', "color": 'mediumslateblue', "hatch": '|'},
    61: {"name": 'Devonian Ls', "color": 'cornflowerblue', "hatch": ''},
    54: {"name": 'Dewey Lake', "color": 'lightgrey', "hatch": '-'},
    118: {"name": 'Grayburg San Andres', "color": 'mintcream', "hatch": ''},
    119: {"name": 'Mississippian Ls Woodford Sh', "color": 'bisque', "hatch": 'o'},
    120: {"name": 'Salado Transil Evaporites', "color": 'lemonchiffon', "hatch": '-'},
    57: {"name": 'Spraberry', "color": 'olive', "hatch": '+'},
    121: {"name": 'Strawn Horseshoe Atol', "color": 'linen', "hatch": '0'},
    58: {"name": 'Wolfcamp', "color": 'saddlebrown', "hatch": '*'},
    0: {"name": 'Unknown', "color": 'white', "hatch": '\\'}
}

def get_formation_patterns(formation_code):
    """Obtains formation patterns.

    Obtains plotting patterns for different formations.

    Arguments:
        formation_code (int): Code for various formation patterns
    """
    pattern = FORMATION_PATTERNS.get(formation_code)
    return pattern['color'], pattern['hatch']

def plot_formation_bars(axes, depths, formations):
    """Plots formation bars.

    Plots formation patterns as a function of depth.

    Arguments:
        axes (TODO): Axes handle at which formation patterns are plotted.

        depths (numpy.array): Array of depths

        formations (numpy.array): Array consisting of formation codes
    """
    min_depth, max_depth = min(depths), max(depths)
    
    unique_formations = np.unique(formations)
    formation_tops = [np.where(formations == formation)[0][-1] for formation in unique_formations]
    
    formation_depths = sorted([depths[formation] for formation in formation_tops])
    formation_ids = [formations[index] for index in formation_tops]

    for i, depth in enumerate(formation_depths):
    
        color, hatch = get_formation_patterns(formation_ids[i])
        
        if i == 0:
            axes.bar(0, np.linspace(min_depth, formation_depths[0], 2), bottom=min_depth, color=color, edgecolor='black', hatch=hatch)
        elif i == len(formation_depths) - 1:
            axes.bar(0, np.linspace(formation_depths[i], max_depth, 2), bottom=formation_depths[i-1], color=color, edgecolor='black', hatch=hatch)
        else:
            axes.bar(0, np.linspace(formation_depths[i], formation_depths[i+1], 2), bottom=formation_depths[i-1], color=color, edgecolor='black', hatch=hatch)
    
    axes.invert_yaxis()
    axes.set_ylim([max(depths), min(depths)])
    axes.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axes.set_title('Formation', fontsize=20, y=1.02)
    axes.yaxis.set_ticklabels([])
    axes.minorticks_on()
    axes.grid(which='y', axis='both', linestyle='-', color='black')
    axes.grid(which='y', axis='both', linestyle='-')
    axes.autoscale(enable=True, axis='x', tight=True)
    
    return axes

def plot_curves(ax, depths, curve, curve_title=None, color='b', tick_labels=False):
    """Plots curves given an axis handle.

    Plots curves as a function of depth along an axis handle.

    Arguments:
        ax (TODO): Axes handle at which curve is plotted.

        depths (numpy.array): Array of depths

        curve (numpy.array): Curve values at depths

        curve_title (str, optional): Title of the plot

        color (str, optional): Color with which the curve is plotted.

        tick_labels (bool, optional): TODO
    """
    assert(len(depths) == len(curve))

    min_depth, max_depth = min(depths), max(depths)

    ax.plot(curve, depths, color=color)

    ax.xaxis.tick_top()
    ax.invert_yaxis()
    ax.set_ylim([max_depth, min_depth])

    if tick_labels:
        ax.yaxis.set_ticklabels([])

    ax.tick_params(axis='both', labelsize=15)
    ax.set_title(curve_title, fontsize=20, y=1.02)
    ax.minorticks_on()
    ax.grid(which='major', axis='both', linestyle='-', color='black')
    ax.grid(which='minor', axis='both', linestyle='-')

    return ax


def plot_composite(plot_curves=PLOT_CURVES, figure_size=(21,21), **kwargs):
    
    depth = kwargs['Depth']
    min_depth, max_depth = min(depth), max(depth)
    
    for curve in plot_curves:
        if curve not in kwargs.keys():
            error_message = curve + ' not in the kwargs dictionary.'
            raise ValueError(error_message)
    
    if isinstance(figure_size, int):
        figure_size = (figure_size, figure_size)
    
    figsize = figure_size
    
    min_depth, max_depth = min(depth), max(depth)
    figure = plt.figure(figsize=figsize)
    
    for i, curve in enumerate(plot_curves):

        if i==0:
            tick=True
        else:
            tick=False

        ax = figure.add_subplot(1, len(plot_curves), i+1)
        
        if curve == 'Formation':
            ax = plot_formation_bars(ax, depths, kwargs['Formation'])
            continue
        else:
            ax = plot_curves(ax, depths, kwargs[curve], curve_title=curve, color='b', tick_labels=tick)
    
    plt.subplots_adjust(wspace=0.05, hspace=0.0)
    
    return figure    


