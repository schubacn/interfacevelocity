

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.mlab import griddata


def make_map(x_array, 
             y_array, 
             value_array, 
             point_spacing=25, 
             cmap='viridis',
             title='Color_Map',
             scale_min=None,
             scale_max=None,
             save_location=None):
    ''' Generates a 2-D colormap of value_array, with coordincates according to 
    x_array and y_array
    
    x_array, y_array, and value_array should be indexed identically
    
    Requires:
        from matplotlib.mlab import griddata
    
    Inputs:
        x_array: an array-like with the x_values of each point
        y_array: an array-like with the y_values of each point
        value_array: an array-like with values to plot at each point
        point_spacing: a number that controls the distance between points in the 
            final output.  Higher numbers give lower resolution.  
        cmap: a string with a valid matplotlib cmap
        title: string title to display on top of the map
        scale_min: lower bound for the color map
        scale_max: upper bound for the color map
        save_location: string filename to output, if desired
    '''
    
    xsteps = int((max(x_array) - min(x_array)) / point_spacing) # resolution in x
    ysteps = int((max(y_array) - min(y_array)) / point_spacing) # resolution in y
    xi = np.linspace(min(x_array), max(x_array), xsteps)
    yi = np.linspace(min(y_array), max(y_array), ysteps)
    grid = griddata(x_array, y_array, value_array, xi, yi, interp='linear') 
    
    fig = plt.figure()
    fig.set_size_inches(30,16)
    ax = fig.gca()
    plt.pcolormesh(xi, yi, grid, cmap=cmap, vmin=scale_min, vmax=scale_max)   
    plt.colorbar()
    ax.set_title(title, fontsize=30, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)
    if save_location:
        plt.savefig(save_location)
    plt.show()