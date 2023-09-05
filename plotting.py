import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px
import argparse

import utils

# argparse correction for boolean values. From https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse/43357954#43357954
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def arg_parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ## Forecasting
    parser.add_argument('--data_file', type=str, default=None, help='file containing data to train & test model')
    parser.add_argument('--start', type=str, default=None, help='Start Date of data to plot, inclusive. Format: %Y-%m-%d Default starts at beginning')
    parser.add_argument('--end', type=str, default=None, help='Finish Date of data to plot, inclusive. Default finishes at end')
    parser.add_argument('--type', type=str, default='plt', help='Type of plotting to use: seaborn, plt, plotly')
    parser.add_argument('--plot', type=str2bool, default=True, help="Plot results")
    parser.add_argument('--title', type=str, default='', help='Title of plot')
    parser.add_argument('--feature', type=str, default=None, help='Which feature to plot. Default plots all features')
    
    parser.add_argument('--interval_file', type=str, default="./processed/comsol2015/combined_1day_crop.csv", help='file containing data to train & test model')
    parser.add_argument('--bins', type=str, default=1000, help='Number of bins to use for each feature in histogram. Can be a single value, or multiple values separated by columns')
    parser.add_argument('--min_duration', type=float, default=1, help='Minimum license duration (s) for histogram')
    parser.add_argument('--max_duration', type=float, default=180000, help='Maximum license duration (s) for histogram')

    parser.add_argument('--scatter', type=str2bool, default=False, help="Scatter plot license usage times")




    return parser.parse_args()

def get_line_graph(data, title="", label="", plt_type = 'plt', plot_show=True, plot_save=False, subplots=False, _set_label=True):

    if plt_type == "seaborn":
        ax = sns.lineplot(data)
        ax.set_ylabel('Number License Check-outs')
        ax.set_xlabel('Time')
        #ax.set_xticklabels([datetime.fromtimestamp(tm) for tm in ax.get_xticks()], rotation=50)
        ax.set_title(f"{title} License Check Outs")
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        if plot_save:
            plt.savefig('./plots/'+title+'_license_line_plot.png')
        if plot_show:
            plt.show()
    elif plt_type == "plt":
        ax = data.plot(drawstyle='steps-post', label=label)
        plt.ylabel('Number License Check-outs')
        plt.xlabel('Time')
        #plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        if plot_save:
            plt.savefig('./plots/'+title+'_license_line_plot.png')
        if plot_show:
            plt.show()
    else:
        #data = data("dateTime")
        fig = px.line(data, title=f"{title} License Check Outs",  template="seaborn",
                labels = {
                    "value" : "Number of Check-outs"
                })
        #ax.set_ylabel('Number of checked out licenses')
        #ax.set_xlabel('time')
        #plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        fig.show()

def get_line_graph_multi(data_list, title="", labels=[], plot_save=False, plot_type="plt"):
    if plot_type=="plt":
        fig, ax = plt.subplots()
        
        for data, label in zip(data_list, labels):
            data.plot(drawstyle='steps-post', label=label, ax=ax)
            plt.ylabel('Number of checked out licenses')
            plt.xlabel('time')
    
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        ax.legend(labels, loc='upper left')
        if plot_save:
            plt.savefig('./plots/'+title+'_license_line_plot.png')
        plt.show()
    else:
        #data = data("dateTime")
        fig = px.line(data, title=f"{title} License Check Outs",  template="seaborn",
                labels = {
                    "value" : "Number of Check-outs"
                })
        #ax.set_ylabel('Number of checked out licenses')
        #ax.set_xlabel('time')
        #plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        fig.show()

def histogram(df: pd.DataFrame, feature_names: list, 
              plot_type='plt', min_val=0, max_val=180000, bins=1000, 
              interval_length_column='delta', feature_name_column='feature_names', 
              show_plot=True):
    
    if type(bins) == int:
        bins = [bins]
    
    while len(bins) < len(feature_names):
        bins.append(bins[-1])
    for f,b in zip(feature_names,bins):
        data = df.query(f"`{feature_name_column}` == @f and @min_val <= `{interval_length_column}` <= @max_val")
        #i = df[feature_name_column] == f && df.loc[:, interval_length_column].between(min_val, max_val)
        # hack to set title name in DataFrame.hist
        data.rename(columns={interval_length_column:f}, inplace=True)
        print(data)
        data.hist(column=f, bins=b)
        plt.ylabel('Occurences')
        plt.xlabel('Time (s)')
    plt.show()

def pdf_fit(df: pd.DataFrame, feature_names: list, 
              plot_type='plt', min_val=0, max_val=180000, bins=1000, 
              interval_length_column='delta', feature_name_column='feature_names', 
              show_plot=True):
    import distfit
    if type(bins) == int:
        bins = [bins]
    
    while len(bins) < len(feature_names):
        bins.append(bins[-1])
    for f,b in zip(feature_names,bins):
        data = df.query(f"`{feature_name_column}` == @f and @min_val <= `{interval_length_column}` <= @max_val")
        #i = df[feature_name_column] == f && df.loc[:, interval_length_column].between(min_val, max_val)
        # hack to set title name in DataFrame.hist
        #data.rename(columns={interval_length_column:f}, inplace=True)
        X = data[interval_length_column].values
        print(X[0:10])
        dfit = distfit.distfit()
        dfit.fit_transform(X)
        print(dfit.summary)
        dfit.plot_summary()
        dfit.plot()        
        plt.ylabel('Occurences')
        plt.xlabel('Time (s)')
    plt.show()

def plot_license_duration(df:pd.DataFrame, features:list, interval_length_column='delta', feature_name_column='feature_names', date_column='check_out'):
        for f in features:
            data = df[df[feature_name_column] == f]
            data.plot.scatter(x=date_column, y=interval_length_column , label=f"{f} License duration")
            plt.ylabel('License usage duration (s)')
            plt.xlabel('Check-out time')
            plt.xticks(rotation=30, horizontalalignment='right')

        plt.show()

if __name__ == "__main__":  
    args = arg_parse()    

    #print(f"current directory: {os.getcwd()}")
    #file = "./processed/comsol2015/combined_1day.csv"#"./data/comsol2015.csv"   
    #file = "./data/comsol2015_denied.csv"#"./data/comsol2015.csv"  
    #name = "Comsol_denied" # should not contain spaces
    title = args.title
    
    #data = pd.read_csv(file, header=0, parse_dates=[2,3]).sort_values('check_out')

    #data = pd.read_csv(file, index_col=0).sort_values('check_out')#sort_index["check_out"] # data must be sorted
    
    if args.data_file != None:
        data = pd.read_csv(args.data_file, index_col=0, parse_dates=[0])
        data = utils.crop_data(data, args.start, args.end)
        
        if args.feature != None:
            features = args.feature.split(',')
            data = data[features]
        get_line_graph(data, title=title, plt_type=args.type)
    
    if args.interval_file != None:
        data = pd.read_csv(args.interval_file, parse_dates=[2,3])
        data = utils.crop_data(data, args.start, args.end, column_name='check_out')
        features = args.feature.split(',')
        bins = [int(x) for x in args.bins.split(',')]        

        #
        histogram(data, features, min_val=args.min_duration, max_val=args.max_duration, bins=bins)
        use_pdf_fit = False
        if use_pdf_fit:
            pdf_fit(data, features, min_val=args.min_duration, max_val=args.max_duration, bins=bins)

        if args.scatter:
            plot_license_duration(data, features)

    

