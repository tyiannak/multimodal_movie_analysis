"""
This script is used to plot histograms of features. 

Usage example:

python3 feature_selection.py -v dataset/Aerial dataset/Tilt

"""

import plotly
import argparse
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.subplots
from train import feature_extraction


def parse_arguments():
    """Parse arguments for real time demo.
    """
    parser = argparse.ArgumentParser(description="Create Shot "
                                                 "Classification Dataset")
    parser.add_argument("-v", "--videos_path", required=True, action='append',
                        nargs='+', help="Videos folder path")


    return parser.parse_args()

def plot_feature_histograms(list_of_feature_mtr, feature_names,
                            class_names, n_columns=5):
    '''
    Plots the histograms of all classes and features for a given
    classification task.
    :param list_of_feature_mtr: list of feature matrices
                                (n_samples x n_features) for each class
    :param feature_names:       list of feature names
    :param class_names:         list of class names, for each feature matr
    '''
    n_features = len(feature_names)
    n_bins = 12
    n_rows = int(n_features / n_columns) + 1
    figs = plotly.subplots.make_subplots(rows=n_rows, cols=n_columns,
                                      subplot_titles=feature_names)
    figs['layout'].update(height=(n_rows * 250))
    clr = get_color_combinations(len(class_names))

    for i in range(n_features):
        # for each feature get its bin range (min:(max-min)/n_bins:max)
        f = np.vstack([x[:, i:i + 1] for x in list_of_feature_mtr])
        np.seterr(all='raise')
        try:
            bins = np.arange(f.min(), f.max(), (f.max() - f.min()) / n_bins)
        except:
            print ('Floating point error')
        for fi, f in enumerate(list_of_feature_mtr):
            # load the color for the current class (fi)
            mark_prop = dict(color=clr[fi], line=dict(color=clr[fi], width=3))
            # compute the histogram of the current feature (i) and normalize:
            h, _ = np.histogram(f[:, i], bins=bins)
            h = h.astype(float) / h.sum()
            cbins = (bins[0:-1] + bins[1:]) / 2
            scatter_1 = go.Scatter(x=cbins, y=h, name=class_names[fi],
                                   marker=mark_prop, showlegend=(i == 0))
            # (show the legend only on the first line)
            figs.append_trace(scatter_1, int(i/n_columns)+1, i % n_columns+1)
    for i in figs['layout']['annotations']:
        i['font'] = dict(size=10, color='#224488')
        
    plotly.offline.plot(figs, filename="report.html", auto_open=True)

def get_color_combinations(n_classes):
    clr_map = plt.cm.get_cmap('jet')
    range_cl = range(int(int(255/n_classes)/2), 255, int(255/n_classes))
    clr = []
    for i in range(n_classes):
        clr.append('rgba({},{},{},{})'.format(clr_map(range_cl[i])[0],
                                              clr_map(range_cl[i])[1],
                                              clr_map(range_cl[i])[2],
                                              clr_map(range_cl[i])[3]))
    return clr


def data_preparation(x, fname):
    """
    Prepare the data before the plot process
    :param x: exracted features from videos
    :fname: names of features
    :return: features, features names, class names
    """
    features = []
    class_names = []
    #Save features and labels to list of numpy arrays
    for key, value in x.items():
        features.append(value)
        class_names.append(key)
    

    #Insert features names to numpy array
    values = fname.values()
    value_iterator = iter(values)
    fnames = next(value_iterator)

    #Add a number to the features for easier detection
    numbers = np.arange(244)
    numbers = numbers.astype(str)
    fnames = np.core.defchararray.add(numbers, fnames)

    return features, fnames, class_names

def remove_features(features,fnames):
    '''
    #COlors + hsv 
    delete = list(range(0,45,1))
    delete.extend(range(52,97,1))
    delete.extend(range(104,149,1))
    delete.extend(range(156,201,1))  
    '''
    #Remove frontal_faces
    #delete = [46,47,98,99,150,151,202,203] 
    
    #Remove object detection features
    delete = list(range(208,243,1))

    for i in delete:
        print (i)

    deleted_features=[]
    for x in features:
        deleted_features.append(np.delete(x,delete,axis=1))
        print(x.shape)
    for x in deleted_features:
        print(x.shape)

    deleted_fnames = np.delete(fnames,delete,axis=0)

    print(deleted_fnames.shape)

    return deleted_features, deleted_fnames

if __name__ == "__main__": 

    args = parse_arguments()
    videos_path = args.videos_path

    # Convert list of lists to a single list
    videos_path = [item for sublist in videos_path for item in sublist]

    #Feature extraction process
    x, _, f_names = feature_extraction(videos_path)

    #Prepare data for plot
    features, fnames, class_names = data_preparation(x,f_names)
    
    features,fnames = remove_features(features,fnames)

    #Plot features histogram   
    plot_feature_histograms(features, fnames, class_names)

