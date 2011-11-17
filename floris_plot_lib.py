# General plotting functions used in making Dickinsonian figures
# written by Floris van Breugel, with some help from Andrew Straw and Will Dickson

# general imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

# used for colorline
from matplotlib.collections import LineCollection

# used in histogram
from scipy.stats import norm as gaussian_distribution
from scipy import signal

# used in colorbar
import matplotlib.colorbar

# not used
#import scipy.optimize
#import scipy.stats.distributions as distributions

###################################################################################################
# Misc Info
###################################################################################################

# FUNCTIONS contained in this file: 
# adjust_spines
# colorline
# histogram
# histogram2d (heatmap)
# boxplot
# colorbar (scale for colormap stuff), intended for just generating a colorbar for use in illustrator figure assembly


# useful links:
# colormaps: http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps


###################################################################################################
# Floris' parameters for saving figures. 
# NOTE: this could mess up your default matplotlib setup, but it allows for saving to pdf
###################################################################################################

from matplotlib import rcParams
fig_width = 3.25 # width in inches
fig_height = 3.25  # height in inches
fig_size =  (fig_width, fig_height)

fontsize = 8
params = {'backend': 'Agg',
          'ps.usedistiller': 'xpdf',
          'ps.fonttype' : 3,
          'pdf.fonttype' : 3,
          'font.family' : 'sans-serif',
          'font.serif' : 'Times, Palatino, New Century Schoolbook, Bookman, Computer Modern Roman',
          'font.sans-serif' : 'Helvetica, Avant Garde, Computer Modern Sans serif',
          'font.cursive' : 'Zapf Chancery',
          'font.monospace' : 'Courier, Computer Modern Typewriter',
          'font.size' : fontsize,
          'text.fontsize': fontsize,
          'axes.labelsize': fontsize,
          'axes.linewidth': 1.0,
          'xtick.major.linewidth': 1,
          'xtick.minor.linewidth': 1,
          #'xtick.major.size': 6,
          #'xtick.minor.size' : 3,
          'xtick.labelsize': fontsize,
          #'ytick.major.size': 6,
          #'ytick.minor.size' : 3,
          'ytick.labelsize': fontsize,
          'figure.figsize': fig_size,
          'figure.dpi' : 72,
          'figure.facecolor' : 'white',
          'figure.edgecolor' : 'white',
          'savefig.dpi' : 300,
          'savefig.facecolor' : 'white',
          'savefig.edgecolor' : 'white',
          'figure.subplot.left': 0.2,
          'figure.subplot.right': 0.8,
          'figure.subplot.bottom': 0.25,
          'figure.subplot.top': 0.9,
          'figure.subplot.wspace': 0.0,
          'figure.subplot.hspace': 0.0,
          'lines.linewidth': 1.0,
          'text.usetex': True, 
          }
rcParams.update(params) 

###################################################################################################
# Adjust Spines (Dickinson style, thanks to Andrew Straw)
###################################################################################################

# NOTE: smart_bounds is disabled (commented out) in this function. It only works in matplotlib v >1.
# to fix this issue, try manually setting your tick marks (see example below) 
def adjust_spines(ax,spines, color={}, spine_locations={}, smart_bounds=False):
    if type(spines) is not list:
        spines = [spines]
    spine_locations_dict = {'top': 10, 'right': 10, 'left': 10, 'bottom': 10}
    for key in spine_locations.keys():
        spine_locations_dict[key] = spine_locations[key]
        
    if 'none' in spines:
        for loc, spine in ax.spines.iteritems():
            spine.set_color('none') # don't draw spine
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks([])
        return
    
    for loc, spine in ax.spines.iteritems():
        if loc in spines:
            spine.set_position(('outward',spine_locations_dict[loc])) # outward by x points
            
            if loc in color.keys():
                c = color[loc]
            else:
                c = 'black'
            
            spine.set_color(c)
            #spine.set_smart_bounds(smart_bounds)
            if loc == 'bottom' and smart_bounds:
                pass#spine.set_smart_bounds(True)
        else:
            spine.set_color('none') # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    elif 'right' in spines:
        ax.yaxis.set_ticks_position('right')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])    
        
    for line in ax.get_xticklines() + ax.get_yticklines():
        #line.set_markersize(6)
        line.set_markeredgewidth(1)
        
def adjust_spines_example():
    
    x = np.linspace(0,100,100)
    y = x**2
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x,y)
    adjust_spines(ax, ['left', 'bottom'])
    fig.savefig('adjust_spines_example.pdf', format='pdf')
    
    
def adjust_spines_example_with_custom_ticks():

    x = np.linspace(0,100,100)
    y = x**2
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x,y)
    
    # NOTE: adjust_spines comes before setting limits and custom ticks!
    adjust_spines(ax, ['left', 'bottom'])
    
    # set limits
    ax.set_xlim(0,100)
    ax.set_ylim(0,20000)
    
    # set custom ticks and tick labels
    xticks = [0, 10, 25, 50, 71, 100] # custom ticks, should be a list
    xticklabels = [str(tick) for tick in xticks]
    # alternatively: xticklabels = ['1st', '2nd', '3rd', '4th', '5th', '6th'], for example
    
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels) 
    
    ax.set_xlabel('x axis, custom ticks\ncoooool!')
    
    fig.savefig('adjust_spines_example.pdf', format='pdf')



###################################################################################################
# Colorline
###################################################################################################

# plot a line in x and y with changing colors defined by z, and optionally changing linewidths defined by linewidth
def colorline(ax, x,y,z,linewidth=1, colormap='jet', norm=None, zorder=1, alpha=1, linestyle='solid'):
        cmap = plt.get_cmap(colormap)
        
        if type(linewidth) is list or np.array:
            linewidths = linewidth
        else:
            linewidths = np.ones_like(z)*linewidth
        
        if norm is None:
            norm = plt.Normalize(np.min(z), np.max(z))
        else:
            norm = plt.Normalize(norm[0], norm[1])
        
        '''
        if self.hide_colorbar is False:
            if self.cb is None:
                self.cb = matplotlib.colorbar.ColorbarBase(self.ax1, cmap=cmap, norm=norm, orientation='vertical', boundaries=None)
        '''
            
        # Create a set of line segments so that we can color them individually
        # This creates the points as a N x 1 x 2 array so that we can stack points
        # together easily to get the segments. The segments array for line collection
        # needs to be numlines x points per line x 2 (x and y)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create the line collection object, setting the colormapping parameters.
        # Have to set the actual values used for colormapping separately.
        lc = LineCollection(segments, linewidths=linewidths, cmap=cmap, norm=norm, zorder=zorder, alpha=alpha, linestyles=linestyle )
        lc.set_array(z)
        lc.set_linewidth(linewidth)
        
        ax.add_collection(lc)

def colorline_example():
    
    def tent(x):
        """
        A simple tent map
        """
        if x < 0.5:
            return x
        else:
            return -1.0*x + 1
    
    pi = np.pi
    t = np.linspace(0, 1, 200)
    y = np.sin(2*pi*t)
    z = np.array([tent(x) for x in t]) 
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # standard colorline
    colorline(ax,t,y,z)
    
    # colorline with changing widths, shifted in x
    colorline(ax,t+0.5,y,z,linewidth=z*5)
    
    # colorline with points, shifted in x
    colorline(ax,t+1,y,z, linestyle='dotted')
    
    # set the axis to appropriate limits
    adjust_spines(ax, ['left', 'bottom'])
    ax.set_xlim(0,2)
    ax.set_ylim(0,1.5)
       
    fig.savefig('colorline_example.pdf', format='pdf')
    
    
###################################################################################################
# Histograms
###################################################################################################
    
# first some helper functions
def custom_hist_rectangles(hist, leftedges, width, facecolor='green', edgecolor='none', alpha=1):
    if type(width) is not list:
        width = [width for i in range(len(hist))]
    rects = [None for i in range(len(hist))]
    for i in range(len(hist)):
        rects[i] = patches.Rectangle( [leftedges[i], 0], width[i], hist[i], facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)
    return rects

def bootstrap_histogram(xdata, bins, normed=False, n=None, return_raw=False):
    if type(xdata) is not np.ndarray:
        xdata = np.array(xdata)

    if n is None:  
        n = len(xdata)
    hist_list = np.zeros([n, len(bins)-1])
    
    for i in range(n):
        # Choose #sample_size members of d at random, with replacement
        choices = np.random.random_integers(0, len(xdata)-1, n)
        xsample = xdata[choices]
        hist = np.histogram(xsample, bins, normed=normed)[0].astype(float)
        hist_list[i,:] = hist
        
    hist_mean = np.mean(hist_list, axis=0)
    hist_std = np.std(hist_list, axis=0)
    
    if return_raw:
        return hist_list
    else:
        return hist_mean, hist_std
        
    
def histogram(ax, data_list, bins=10, bin_width_ratio=0.6, colors='green', edgecolor='none', bar_alpha=0.7, curve_fill_alpha=0.4, curve_line_alpha=0.8, curve_butter_filter=[3,0.3], return_vals=False, show_smoothed=True, normed=False, normed_occurences=False, bootstrap_std=False, bootsrap_line_width=0.5, exponential_histogram=False):
    
    n_bars = float(len(data_list))
    if type(bins) is int:
    
        mia = np.array([np.min(d) for d in data_list])
        maa = np.array([np.max(d) for d in data_list])
        
        bins = np.linspace(np.min(mia), np.max(maa), bins, endpoint=True)
        
    if type(colors) is not list:
        colors = [colors]
    if len(colors) != n_bars:
        colors = [colors[0] for i in range(n_bars)]
        
    bin_centers = np.diff(bins)/2. + bins[0:-1]
    bin_width = np.mean(np.diff(bins))
    bin_width_buff = (1-bin_width_ratio)*bin_width/2.
    bar_width = (bin_width-2*bin_width_buff)/n_bars
    
    butter_b, butter_a = signal.butter(curve_butter_filter[0], curve_butter_filter[1])
    
    if return_vals:
        data_hist_list = []
        data_curve_list = []
        data_hist_std_list = []
        
    # first get max number of occurences
    max_occur = []
    for i, data in enumerate(data_list):
        data_hist = np.histogram(data, bins=bins, normed=normed)[0].astype(float)
        max_occur.append(np.max(data_hist))
    max_occur = np.max(np.array(max_occur))
        
    for i, data in enumerate(data_list):
        
        if bootstrap_std:
            data_hist, data_hist_std = bootstrap_histogram(data, bins=bins, normed=normed)
        else:
            data_hist = np.histogram(data, bins=bins, normed=normed)[0].astype(float)
            
        if exponential_histogram:
            data_hist = np.log(data_hist)
        
        if normed_occurences is not False:
            if normed_occurences == 'total':
                data_hist /= max_occur 
                if bootstrap_std:
                    data_hist_std /= max_occur
            else:
                div = float(np.max(data_hist))
                print div
                data_hist /= div 
                if bootstrap_std:
                    data_hist_std /= div
                    
        
        rects = custom_hist_rectangles(data_hist, bins[0:-1]+bar_width*i+bin_width_buff, width=bar_width, facecolor=colors[i], edgecolor=edgecolor, alpha=bar_alpha)
        if bootstrap_std:
            for j, s in enumerate(data_hist_std):
                x = bins[j]+bar_width*i+bin_width_buff + bar_width/2.
                #ax.plot([x,x], [data_hist[j], data_hist[j]+data_hist_std[j]], alpha=1, color='w')
                ax.plot([x,x], [data_hist[j], data_hist[j]+data_hist_std[j]], alpha=bar_alpha, color=colors[i], linewidth=bootsrap_line_width)
                
                #ax.plot([x-bar_width/3., x+bar_width/3.], [data_hist[j]+data_hist_std[j],data_hist[j]+data_hist_std[j]], alpha=1, color='w')
                #ax.plot([x-bar_width/3., x+bar_width/3.], [data_hist[j]+data_hist_std[j],data_hist[j]+data_hist_std[j]], alpha=bar_alpha, color=colors[i])
        for rect in rects:
            rect.set_zorder(1)
            ax.add_artist(rect)
        
                
        if show_smoothed:
            data_hist_filtered = signal.filtfilt(butter_b, butter_a, data_hist)
            interped_bin_centers = np.linspace(bin_centers[0]-bin_width/2., bin_centers[-1]+bin_width/2., 100, endpoint=True)
            v = 100 / float(len(bin_centers))
            interped_data_hist_filtered = np.interp(interped_bin_centers, bin_centers, data_hist_filtered)
            interped_data_hist_filtered2 = signal.filtfilt(butter_b/v, butter_a/v, interped_data_hist_filtered)
            #ax.plot(bin_centers, data_hist_filtered, color=facecolor[i])
            if curve_fill_alpha > 0:
                ax.fill_between(interped_bin_centers, interped_data_hist_filtered2, np.zeros_like(interped_data_hist_filtered2), color=colors[i], alpha=curve_fill_alpha, zorder=-100, edgecolor='none')
            if curve_line_alpha:
                ax.plot(interped_bin_centers, interped_data_hist_filtered2, color=colors[i], alpha=curve_line_alpha)
        
        if return_vals:
            data_hist_list.append(data_hist)
            if bootstrap_std:
                data_hist_std_list.append(data_hist_std)
            
            if show_smoothed:
                data_curve_list.append(data_hist_filtered)
                
    if return_vals and bootstrap_std is False:
        return bins, data_hist_list, data_curve_list
    elif return_vals and bootstrap_std is True:
        return bins, data_hist_list, data_hist_std_list, data_curve_list
    
def histogram_example():
    
    # generate a list of various y data, from three random gaussian distributions
    y_data_list = []
    for i in range(3):
        mean = np.random.random()*10
        std = 3
        ndatapoints = 500
        y_data = gaussian_distribution.rvs(loc=mean, scale=std, size=ndatapoints)
        y_data_list.append(y_data)
        
    nbins = 40
    bins = np.linspace(-10,30,nbins)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    histogram(ax, y_data_list, bins=bins, bin_width_ratio=0.8, colors=['green', 'black', 'orange'], edgecolor='none', bar_alpha=1, curve_fill_alpha=0.4, curve_line_alpha=0, curve_butter_filter=[3,0.3], return_vals=False, show_smoothed=True, normed=True, normed_occurences=False, bootstrap_std=False, exponential_histogram=False)
    
    adjust_spines(ax, ['left', 'bottom'])
    
    fig.savefig('histogram_example.pdf', format='pdf')
    
###################################################################################################
# Boxplots
###################################################################################################
    
def boxplot(ax, x_data, y_data_list, nbins=50, colormap='YlOrRd', colorlinewidth=2, boxwidth=1, usebins=None, boxlinewidth=0.5, outlier_limit=0.01, norm=None, use_distribution_for_linewidth=False):    

    if usebins is None: 
        usebins = nbins
        # usebins lets you assign the bins manually, but it's the same range for each x_coordinate

    for i, y_data in enumerate(y_data_list):
        #print len(y_data)
    
        # calc boxplot statistics
        median = np.median(y_data)
        ind = np.where(y_data<=median)[0].tolist()
        first_quartile = np.median(y_data[ind])
        ind = np.where(y_data>=median)[0].tolist()
        last_quartile = np.median(y_data[ind])
        #print first_quartile, median, last_quartile
        
        # find outliers
        ind_sorted = np.argsort(y_data)
        bottom_limit = int(len(ind_sorted)*(outlier_limit))
        top_limit = int(len(ind_sorted)*(1-outlier_limit))
        indices_inrange = ind_sorted[bottom_limit:top_limit]
        outliers = ind_sorted[0:bottom_limit].tolist() + ind_sorted[top_limit:len(ind_sorted)-1].tolist()
        y_data_inrange = y_data[indices_inrange]
        y_data_outliers = y_data[outliers]
    
    
        # plot colorline
        x = x_data[i]
        hist, bins = np.histogram(y_data_inrange, usebins, normed=False)
        hist = hist.astype(float)
        hist /= np.max(hist)
        x_arr = np.ones_like(bins)*x
        
        if use_distribution_for_linewidth:
            colorlinewidth = hist*colorlinewidth
        
        colorline(ax, x_arr, bins, hist, colormap=colormap, norm=norm, linewidth=colorlinewidth) # the norm defaults make it so that at each x-coordinate the colormap/linewidth will be scaled to show the full color range. If you want to control the color range for all x-coordinate distributions so that they are the same, set the norm limits when calling boxplot(). 
        
        
        # plot boxplot
        ax.hlines(median, x-boxwidth/2., x+boxwidth/2., color='black', linewidth=boxlinewidth)
        ax.hlines([first_quartile, last_quartile], x-boxwidth/2., x+boxwidth/2., color='black', linewidth=boxlinewidth/2.)
        ax.vlines([x-boxwidth/2., x+boxwidth/2.], first_quartile, last_quartile, color='black', linewidth=boxlinewidth/2.)
        
        # plot outliers
        if outlier_limit > 0:
            x_arr_outliers = x*np.ones_like(y_data_outliers)
            ax.plot(x_arr_outliers, y_data_outliers, '.', markerfacecolor='gray', markeredgecolor='none', markersize=1)
        
    
def boxplot_example():

    # generate a list of various y data, from three random gaussian distributions
    x_data = np.linspace(0,20,5)
    y_data_list = []
    for i in range(len(x_data)):
        mean = np.random.random()*10
        std = 3
        ndatapoints = 500
        y_data = gaussian_distribution.rvs(loc=mean, scale=std, size=ndatapoints)
        y_data_list.append(y_data)
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    boxplot(ax, x_data, y_data_list)
    adjust_spines(ax, ['left', 'bottom'])
    
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    
    fig.savefig('boxplot_example.pdf', format='pdf')    
        
    
    
###################################################################################################
# 2D "heatmap" Histogram
###################################################################################################

def histogram2d(ax, x, y, bins=100, normed=False, histrange=None, weights=None, logcolorscale=False, colormap='jet', interpolation='nearest'):
    # the following paramters get passed straight to numpy.histogram2d
    # x, y, bins, normed, histrange, weights
    
    # from numpy.histogram2d:
    '''
    Parameters
    ----------
    x : array_like, shape(N,)
      A sequence of values to be histogrammed along the first dimension.
    y : array_like, shape(M,)
      A sequence of values to be histogrammed along the second dimension.
    bins : int or [int, int] or array-like or [array, array], optional
      The bin specification:
    
        * the number of bins for the two dimensions (nx=ny=bins),
        * the number of bins in each dimension (nx, ny = bins),
        * the bin edges for the two dimensions (x_edges=y_edges=bins),
        * the bin edges in each dimension (x_edges, y_edges = bins).
    
    range : array_like, shape(2,2), optional
      The leftmost and rightmost edges of the bins along each dimension
      (if not specified explicitly in the `bins` parameters):
      [[xmin, xmax], [ymin, ymax]]. All values outside of this range will be
      considered outliers and not tallied in the histogram.
    normed : boolean, optional
      If False, returns the number of samples in each bin. If True, returns
      the bin density, ie, the bin count divided by the bin area.
    weights : array-like, shape(N,), optional
      An array of values `w_i` weighing each sample `(x_i, y_i)`. Weights are
      normalized to 1 if normed is True. If normed is False, the values of the
      returned histogram are equal to the sum of the weights belonging to the
      samples falling into each bin.
    '''
    
    hist,x,y = np.histogram2d(x, y, bins, normed=False, range=histrange, weights=weights)
    
    if logcolorscale:
        hist = np.log(hist+1) # the plus one solves bin=0 issues
    
    # make the heatmap
    cmap = plt.get_cmap(colormap)
    ax.imshow(  hist.T, 
                cmap=cmap,
                extent=(x[0], x[-1], y[0], y[-1]), 
                origin='lower', 
                interpolation=interpolation)
                
def histogram2d_example():  

    # make some random data
    mean = np.random.random()*10
    std = 3
    ndatapoints = 50000
    x = gaussian_distribution.rvs(loc=mean, scale=std, size=ndatapoints)
    y = gaussian_distribution.rvs(loc=mean, scale=std, size=ndatapoints)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    histogram2d(ax, x, y, bins=100)
    
    adjust_spines(ax, ['left', 'bottom'])
    
    fig.savefig('histogram2d_example.pdf', format='pdf')





###################################################################################################
# Colorbar
###################################################################################################

def colorbar(ax=None, ticks=None, colormap='jet', aspect=20, orientation='vertical', filename=None, flipspine=False):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    if ticks is None:
        ticks = np.linspace(-1,1,5,endpoint=True)
    
    ax.set_aspect('equal')
    
    # horizontal
    if orientation == 'horizontal':
        xlim = (ticks[0],ticks[-1])
        yrange = (ticks[-1]-ticks[0])/float(aspect)
        ylim = (0, yrange)
        grad = np.linspace(ticks[0], ticks[-1], 500, endpoint=True)
        im = np.vstack((grad,grad))
        if not flipspine:
            adjust_spines(ax,['bottom'])
        else:
            adjust_spines(ax,['top'])
    
    # vertical
    if orientation == 'vertical':
        ylim = (ticks[0],ticks[-1])
        xrange = (ticks[-1]-ticks[0])/float(aspect)
        xlim = (0, xrange)
        grad = np.linspace(ticks[0], ticks[-1], 500, endpoint=True)
        im = np.vstack((grad,grad)).T
        if not flipspine:
            adjust_spines(ax,['right'])
        else:
            adjust_spines(ax,['left'])

    # make image
    cmap = plt.get_cmap(colormap)
    ax.imshow(  im, 
                cmap=cmap,
                extent=(xlim[0], xlim[-1], ylim[0], ylim[-1]), 
                origin='lower', 
                interpolation='nearest')
                
    if filename is not None:
        fig.savefig(filename, format='pdf')
    
def colorbar_example():
    colorbar(filename='colorbar_example.pdf')




    
    
