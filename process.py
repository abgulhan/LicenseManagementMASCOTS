import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import warnings
import argparse
import os
warnings.filterwarnings("ignore", category=FutureWarning)



import queue
import multiprocessing as mp



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

    ## Preprocessing
    # log processing
    parser.add_argument('--process_data', type=str2bool, default=False, help='Perform data processing if not done before')
    parser.add_argument('--process_file', type=str, default='./data/comsol2015.csv', help='Data file to process. Should be output of log_parse.py. Required for all parameters')
    parser.add_argument('--process_dir', type=str, default='./processed/comsol2015', help='Directory to store processed features')
    # spacing
    parser.add_argument('--process_even_spacing', type=str2bool, default=False, help='Evenly space out data')
    #   TODO use better format for time
    parser.add_argument('--H', type=int, default=0, help='Even spacing number of hours')
    parser.add_argument('--M', type=int, default=0, help='Even spacing number of minutes')
    parser.add_argument('--S', type=int, default=0, help='Even spacing number of seconds')
    parser.add_argument('--even_spacing_output', type=str, default='./processed/comsol2015/combined_1sec.csv', help='location to save even spaced data')
    parser.add_argument('--process_extension', type=str, default='.csv', help='Extension of processed data')
    # adding additional features
    parser.add_argument('--process_license_duration', type=str2bool, default=False, help='Augments training data with average license duration')
    parser.add_argument('--process_license_duration_window_size', type=str, default='D', help='Window size for taking average license duration. Ex: 1D, 1H, 5M, 1M')
    parser.add_argument('--process_license_duration_log_file', type=str, default='./data/comsol2015.csv', help='License duration log file name')
    parser.add_argument('--process_license_duration_output', type=str, default=None, help='Location to save augmented data')
    # extract high denial periods
    parser.add_argument('--process_high_denial_unique', type=str2bool, default='False', help='Find dates with high denial times. Get number of unique users (i.e. ignore resubmissions). Defualt 2D')
    parser.add_argument('--process_high_denial', type=str2bool, default=False, help='Find dates with high denial times')
    parser.add_argument('--process_high_denial_window', type=str, default='2D', help='Window size for finding dates with high denial times. Defualt 2D')
    parser.add_argument('--process_high_denial_threshold', type=int, default=10, help='Minimum number of denials within window')
    parser.add_argument('--process_high_denial_log', type=str, default='./data/comsol2015_denied.csv', help='Location of denied logs (output of log_parse.py).')
    parser.add_argument('--process_high_denial_output', type=str, default=None, help='Save modified denied logs to this location')
    parser.add_argument('--process_high_denial_load', type=str, default=None, help='Load modified denied logs')
    # get license usage statistics
    parser.add_argument('--get_license_usage_statistics', type=str2bool, default='False', help='Find average license utilization')
    parser.add_argument('--duration_file', type=str, default='./data/comsol2015.csv', help='License duration log file name')
    parser.add_argument('--forecast_file', type=str, default='./processed/combined/combined_1day.csv', help='License forecast data file name')
    parser.add_argument('--max_license_file', type=str, default=None, help='Maximum license count file. Output of lmstat')
    parser.add_argument('--max_license_file_type', type=str, default=None, help='Type of max_license_file. Different licenses may have differnt formats')
    parser.add_argument('--skipped_duration', type=int, default=0, help='Length of time period to ignore in statistics calculations. Format: seconds')

    parser.add_argument('--get_peak_license_usage_statistics', type=str2bool, default='False', help='Find average peak license usage')

    # convert to long format, for use in autoGluon
    parser.add_argument('--to_long_format', type=str2bool, default='False', help='Convert processed data to long format.')
    parser.add_argument('--long_format_output', type=str, default=None, help='Location to save long data')

    return parser.parse_args()

# ignore this function. preprocess_max_per_unit_time() is more accurate
def preprocess_total_per_unit_time(input_file="matlab1.csv", output_file="data/matlab_1h.csv"):
    df = pd.read_csv(input_file, header=0, parse_dates=[2,3])
    #print(df)
    print(df.keys)
    print(df.columns.tolist())
    print(df.dtypes)
    next_hour = None

    dtypes = np.dtype(
    [
        ("date", np.datetime64),
        ("num_check_out", int)
    ])
    parsed = pd.DataFrame(np.empty(0, dtype=dtypes))

    rows = len(df.index)

    for index,row in df.iterrows():
        next_hour = row["check_out"].floor('h') + pd.DateOffset(hours=1)
        print(next_hour)
        row = {"date": row["check_out"].floor('h'), "num_check_out": 0}
        parsed = parsed.append(row, ignore_index=True)
        break
    print(parsed)
    i=0
    t=datetime.datetime.now() # start time
    for index, row in df.iterrows():
        if i%10000 == 0:
            print(f"{100*i//rows}%  row {i}, time={datetime.datetime.now()-t}")
        if row["feature_names"] != "MATLAB":
            continue
        current_time = row['check_out']
        if current_time >= next_hour:
            next_hour = row["check_out"].floor('h') + pd.DateOffset(hours=1)
            row = {"date": row["check_out"].floor('h'), "num_check_out": 0}
            parsed = parsed.append(row, ignore_index=True)
        parsed.loc[parsed.index[-1], 'num_check_out'] += 1
        i+=1
    print(parsed)
            
    parsed.to_csv(output_file)

'''
Converts from license usage intervals into licenses used at a given time. 
Note that the time difference between data points varies. Only when there is
a change, a new data point is added. Need to call even_spacing() before this can
be used for forecasting.
'''
# input file must be sorted by check_out!!    
def preprocess_max_per_unit_time(input_file="matlab1.csv", output_file="", feature="MATLAB", column_name="", show_progress=True):
    # helper function
    def head(queue):
        assert(not queue.empty())
        tmp = queue.get()
        queue.put(tmp)
        return tmp 
    print(f"parsing feature {feature}")
    
    output_file_exists = True
    try:
        f = open(output_file)
        f.close()
    except FileNotFoundError:
        output_file_exists = False
    
    if output_file_exists:
        print(f"file for feature {feature} exists, skipping...")
        return True
    
    df = pd.read_csv(input_file, header=0, parse_dates=[2,3]).sort_values('check_out')
    #print(df)
    #print(df.keys)
    #print(df.columns.tolist())
    #print(df.dtypes)

    dtypes = np.dtype(
    [
        ("date", np.datetime64(1,'s')),
        ("num_check_out", int)
    ])
    parsed = pd.DataFrame(np.empty(0, dtype=dtypes))

    rows = len(df.index)
    #print(parsed)
    i=0
    t=datetime.datetime.now() # start time
    
    data = {} # key=dateTime, value = int
    in_queue = queue.PriorityQueue()
    # reformat data file
    ctr = 0
    '''
    for index, row in df.iterrows():
        if row["feature_names"] != feature: # 
            continue
        delta = row['delta']
        if delta == 0:
            continue

        license_check_out = row['check_out']
        license_check_in = row['check_in']
        data[license_check_out] = 1
        in_queue.put(license_check_in)
        break
    '''
    ctr = 0 # count current number of in use licences
    for index, row in df.iterrows():
        if show_progress:
            if i%100000 == 0:
                print(f"{100*i//rows}%  row {i}, feature {feature}, time={datetime.datetime.now()-t}")
            i+=1
        if row["feature_names"] != feature: # 
            continue
        delta = row['delta'] # skip 0 time check outs
        if delta == 0:
            continue

        license_check_out = row['check_out']
        license_check_in = row['check_in']
        in_queue.put(license_check_in)
        future_check_in = head(in_queue)
        if license_check_out == future_check_in: #check out and check in in same second
            in_queue.get()
        elif license_check_out < future_check_in: # no check ins yet
            ctr += 1
            data[license_check_out] = ctr
        else: # license_check_out > future_check_in
            while not in_queue.empty() and head(in_queue) < license_check_out:
                ctr -= 1
                assert (ctr >= 0)
                check_in_time = in_queue.get() # also removes element from queue
                data[check_in_time] = ctr
            # all previous check ins complete, now can check out
            ctr += 1
            data[license_check_out] = ctr
    # end for
    while not in_queue.empty():
        ctr -= 1
        assert (ctr >= 0)
        check_in_time = in_queue.get() # also removes element from queue
        data[check_in_time] = ctr
    #print(ctr)
    assert(ctr==0)
            
    parsed = pd.DataFrame.from_dict(data, orient='index')
    if parsed.empty:
        print(f"Feature {feature} has no entries with delta > 0")
        return parsed
    if column_name != "":
        parsed.columns = [column_name] # index is dateTime
    else:
        parsed.columns = ["max_check_outs"] # index is dateTime
    #print(parsed)
    
    if output_file != "":
        parsed.to_csv(output_file)
    
    return parsed
    
def get_features(input_file):
    df = pd.read_csv(input_file, header=0, parse_dates=[2,3])
    return df['feature_names'].unique()


def even_spacing(data, seconds=1, minutes=0, hours=0, col_name='max_check_outs'):
    if not data.index.is_monotonic_increasing:
        data = data.sort_index()
    new_data = {}
    #col_name = 'max_check_outs'
    first_row = data.iloc[0][col_name]
    first_date = data.index[0]
    
    #round down to day to get consistent results across files
    #TODO Find better way to do this for given increment parameter
    first_date = first_date.floor("D")
    
    last_row = data.iloc[-1][col_name]
    last_date = data.index[-1]

    current_date = first_date
    increment = datetime.timedelta(seconds=seconds, minutes=minutes, hours=hours)
    prev_amount = first_row
    
    feature_name = data.columns[-1]
    
    i = 0
    '''
    # TODO parallelize or make more efficient
    while current_date <= last_date:
        if i%100000==0:
            print(f"progress {last_date-current_date} remaining")
        i += 1
        
        if current_date in data.index:
            new_data[current_date] = data.loc[current_date][col_name]
            prev_amount = data.loc[current_date][col_name]
        else:
            new_data[current_date] = prev_amount
            
        current_date += increment
    '''
    i=0
    max_val_in_interval = 0
    last_val = 0
    while current_date <= last_date:
        if i%100==0:
            print(f"progress {last_date-current_date} remaining for {feature_name}")
        i += 1
        
        interval = data.loc[current_date:current_date+increment, col_name]
        if len(interval) != 0:
            max_val_in_interval = interval.max()
            last_val = interval.loc[interval.index.max()]  

        else: # use last value of preious period
            max_val_in_interval = last_val
        new_data[current_date] = max_val_in_interval

        current_date += increment

    parsed = pd.DataFrame.from_dict(new_data, orient='index')
    parsed.columns = [col_name] # index is dateTime
    
    return parsed



"""
Uses denied logs (output of log_parse.py) to find periods with high denial times.

min_denial_per_period: how many denials to look for in a period
max_job_length: maximum length of a job. Must be time delta object
sort_data: whether to sort input input file data. Note if data is not sorted by denial_time then this must be set to true
col_name: name of datetime column
"""

def identify_high_denial_times(input_file, min_denial_per_period, max_job_length=pd.to_timedelta('2D'), sort=True, col_name='denied_dateTime', new_col_name='denials_past_period', save_dir=None, load_saved_file=False, unique_names = False):
    if load_saved_file:
        if sort:
            data = pd.read_csv(input_file, header=0, parse_dates=[col_name]).sort_values(col_name)
        else:
            data = pd.read_csv(input_file, header=0, parse_dates=[col_name])
    else:
        if sort:
            data = pd.read_csv(input_file, header=0, parse_dates=[col_name]).sort_values(col_name)
        else:
            data = pd.read_csv(input_file, header=0, parse_dates=[col_name])
        
        
        prev_row = data.iloc[0]
        start_date = data.iloc[0][col_name]
        denials_past_interval = []
        if unique_names:
            for i in range(len(data)):
                
                row_date = data.iloc[i][col_name]
                start_date = row_date-max_job_length
                query = data.query(f'{col_name} > @start_date and {col_name} <= @row_date')
                unique = query['user_name'].unique()
                denial_count = len(unique)
                denials_past_interval.append(denial_count)
        else:
            for i in range(len(data)):
                
                row_date = data.iloc[i][col_name]
                start_date = row_date-max_job_length
                denial_count = len(data.query(f'{col_name} > @start_date and {col_name} <= @row_date'))

                denials_past_interval.append(denial_count)
            
        data[new_col_name] = denials_past_interval
        
        if save_dir != None:
            data.to_csv(save_dir, index=False)
    
    
    # TODO make this save to file instead of printing
    df = data.loc[data[new_col_name] >= min_denial_per_period, (col_name, new_col_name)].copy()
    df['denied_date'] = data[col_name].dt.date
    df.sort_values(by=['denied_date'])
    print("date, max_denials")
    for date in df['denied_date'].unique():
        max_denials = df.loc[df['denied_date'] == date, new_col_name].max()
        print(f"{date},  {max_denials}")
    return 

'''
Parallel function for generate_feature_avg_license_duration()
'''
def _parallel_generate_feature_avg_license_duration(pass_list):
    row_dates, window, log_data, sort_col_name = pass_list
    cols = []
    for row_date in row_dates:
        start_date = row_date - window
        #mask = (log_data[sort_col_name] > row_date-window) & (log_data[sort_col_name] <= row_date)
        #avg = float(log_data.loc[mask]['delta'].mean())
        avg = float(log_data.query(f'{sort_col_name} > @start_date and {sort_col_name} <= @row_date')['delta'].mean())

        if np.isnan(avg):
            avg=0
        cols.append(avg)
    return cols

'''
Generate average license usage for past time window for every entry in parameter data.
Takes average of all licenses ending in window.
return modified data

data: input data set for forecasting model. Will be augmented with an additional column. Type Pandas dataframe.
window: Determines how far back to take license usage durations' average
log_interval_file: License interval file name. Output of log_parse.py
col_name: name of new feature column to be added to data
sort_col_name: name of column to sort data by in ascending order. Default = check_in
feat_names: names of license features to take average of. Passing None will use all features TODO implement
'''
def generate_feature_avg_license_duration(data, window, log_interval_file, col_name="avg_duration", sort_col_name="check_in", feat_names=None, cpu_count=8, save_dir=None, show_progress=True):
    import time
    log_data = pd.read_csv(log_interval_file, header=0, parse_dates=[sort_col_name]).sort_values(sort_col_name)
    new_col = []
    chunksize = max((len(data)//cpu_count)//2, 256)
    print(f"chunksize = {chunksize}")
    print(f"cpu count = {cpu_count}")
    pass_list = [(data.index[i:i+chunksize], window, log_data, sort_col_name) for i in range(0, len(data), chunksize)]
    start = time.time()
    
    # TODO why is parallel speedup so low?
    with mp.Pool(processes = cpu_count) as p:
        results = list(p.map(_parallel_generate_feature_avg_license_duration, pass_list))
    new_col_ = [item for sublist in results for item in sublist]
    print(f"mp time =  {time.time()-start}")
    
    '''
    # sequential
    start = time.time()
    k = 0
    prog_interval = len(data)//10
    for i in range(len(data)):
        if k%prog_interval == 0:
            print(f"progress {100*k/len(data)}%  time={time.time()-start}")
        k+=1
        row_date = data.index[i]
        start_date = row_date-window
        #mask = (log_data[sort_col_name] > row_date-window) & (log_data[sort_col_name] <= row_date)
        avg = float(log_data.query(f'{sort_col_name} > @start_date and {sort_col_name} <= @row_date')['delta'].mean())
        #print(log_data.query(f'{sort_col_name} > @start_date and {sort_col_name} < @row_date'))
        if np.isnan(avg):
            avg=0
        new_col.append(avg)
    print(f"time to add averge license duration =  {time.time()-start}s")
    
    '''
    
    '''
    # slowest method
    start = time.time()
    cur_window = []
    j=0
    new_col_ = []
    for i in range(len(data)):
        row_date = data.index[i]
        start_date = row_date-window
        while row_date >= log_data.iloc[j][sort_col_name]: # add dates to end of window
            print(f"{row_date} >= {log_data.iloc[j][sort_col_name]}   j={j}")
            cur_window.append((log_data.iloc[j][sort_col_name], log_data.iloc[j]['delta']))
            j += 1
        while len(cur_window) != 0 and start_date < cur_window[0][0]: # remove dates from beginning of window
            del cur_window[0]
            print(len(cur_window))
        if len(cur_window) == 0:
            avg=0
        else:
            sum=0
            for date, delta in cur_window:
                sum += date
            avg = sum/len(cur_window)
        new_col_.append(avg)
    print(f"method 2 time to add averge license duration =  {time.time()-start}s")
    '''

    comp = False#(new_col_ == new_col)
    
    if not comp:
        for i in range(len(new_col)):
            if new_col[i] != new_col_[i]:
                print(f"index {i} value {new_col[i]} vs {new_col_[i]}")
                
    data[col_name]=new_col_
    
    if save_dir != None:
        data.to_csv(save_dir)
    return data

'''
Get usage statistics for each license in the license duration file
license_interval_fname: name of license interval file - output of log_parse.py
license_forecasr_fname: Optional, used for peak usage. Name of license forecast data - output of process.py
'''    
def get_license_usage_statistics(license_interval_fname, license_forecast_fname = None, max_licenses = None, skipped_time = datetime.timedelta(0)):
    log_data = pd.read_csv(license_interval_fname, header=0, parse_dates=['check_in', 'check_out'])
    feature_names = log_data['feature_names'].unique()
    total_time: datetime.timedelta = log_data['check_in'].max() - log_data['check_out'].min()
    total_seconds = total_time.total_seconds()
    total_seconds -= skipped_time # use in case of missing data in logs

    if license_forecast_fname != None:
        forecast_data = pd.read_csv(license_forecast_fname, index_col=0)
    stats = {}
    for feature in feature_names:
        stats[feature] = {}
        total_usage = log_data[log_data['feature_names'] == feature]['delta'].sum()
        total_usage = int(total_usage) 
        avg_usage = total_usage/total_seconds
        stats[feature]['avg_usage'] = avg_usage
        if max_licenses != None:
            utilization = avg_usage/max_licenses[feature]
            stats[feature]['utilization'] = utilization
        if license_forecast_fname != None:
            try:
                peak_use = int(forecast_data[feature].max())
            except:
                peak_use = 0
            stats[feature]['peak_use'] = peak_use
        
    # add any licenses that appear in max_licenses file, but not in logs
    # TODO make this work without max_licenses (license file)
    for feature in max_licenses.keys():
        if stats.get(feature) == None:
           stats[feature] = {}
           stats[feature]['avg_usage'] = 0
           stats[feature]['utilization'] = 0
           stats[feature]['peak_use'] = 0
            
    # sort license features by utilization
    stats = dict(sorted(stats.items(), key=lambda item: item[1]['utilization'], reverse=True))
    
    for key, value in stats.items():
        name = key
        avg_usage = value['avg_usage']
        utilization = value.get('utilization')
        peak_use = value.get('peak_use')
        print(f"{name:<20}:", end='')
        if max_licenses != None:
            print(f" utilization {utilization*100:.3f}%;", end='')
        print(f" avg. use {avg_usage:.3f};", end='')
        if license_forecast_fname != None:
            print(f" peak use {peak_use};", end='')
        print(f" purchased licenses {max_licenses[name]}")

def get_license_peak_statistics(log_fname, num_days=1642, spacing = 'day'):
    data = pd.read_csv(log_fname, index_col=0, parse_dates=[0])
    license_names=list(data.columns)
    for license in license_names:
        avg_peak = data[license].sum()/num_days
        std_peak = data[license].std()
        
        print(f"License {license} average peak usage per {spacing} = {avg_peak}, std {std_peak}")

'''
Convert processed data into a long format, for use in autoGluon

processed_files list of tuples of file_name,feature_name

returns DataFrame
'''
def to_long_format(processed_files):
    long_data = pd.DataFrame(columns=['item_id', 'timestamp', 'target'])
    for file, feature in processed_files:
        df = pd.read_csv(file, parse_dates=[0])
        df = df.rename(columns={df.columns[0]: 'timestamp', df.columns[1]: 'target'})
        df['item_id'] = feature
        long_data = pd.concat([long_data, df], axis=0)
    return long_data

# parallel function
def parallel_process(pass_list):
    input_file, output_file, feature = pass_list
    return preprocess_max_per_unit_time(input_file=input_file, output_file=output_file, 
                                                    feature=feature, show_progress=True, column_name=feature)
    
def parallel_even_spaceing(pass_list):
    input_file, col_name, interval = pass_list
    seconds, minutes, hours = interval
    print(f"processing {col_name}")
    return even_spacing(pd.read_csv(input_file, index_col=0, parse_dates=[0]), col_name=col_name, seconds=seconds, minutes=minutes, hours=hours)

if __name__ == '__main__':
    args = arg_parse()    
    
    
    if args.get_peak_license_usage_statistics:
        peak_log_file = args.forecast_file
        get_license_peak_statistics(peak_log_file)
        
    if args.get_license_usage_statistics:
        duration_file = args.duration_file
        forecast_file = args.forecast_file
        max_license_file = args.max_license_file
        max_licenses = None
        if max_license_file != None:
            import utils
            max_license_file_type = args.max_license_file_type 
            max_licenses = utils.parse_max_licenses(max_license_file, license_type=max_license_file_type, multiple_license_versions_behavior='max') #TODO auto type detect
            print(max_licenses)
        skipped_duration =  args.skipped_duration
        get_license_usage_statistics(duration_file, forecast_file, max_licenses=max_licenses, skipped_time=skipped_duration)
    
    if args.process_high_denial_unique:
        w = args.process_high_denial_window
        window_size = pd.to_timedelta(w if w[0].isdecimal() else '1' + w)
        denials = identify_high_denial_times(input_file=args.process_high_denial_log, 
                                   min_denial_per_period=args.process_high_denial_threshold,
                                   max_job_length=window_size, 
                                   save_dir=args.process_high_denial_output,
                                   sort=True,
                                   col_name='denied_dateTime',
                                   new_col_name='denials_past_period',
                                   load_saved_file=args.process_high_denial_load,
                                   unique_names=True
                                   )
    if args.process_high_denial:
        w = args.process_high_denial_window
        window_size = pd.to_timedelta(w if w[0].isdecimal() else '1' + w)
        denials = identify_high_denial_times(input_file=args.process_high_denial_log, 
                                   min_denial_per_period=args.process_high_denial_threshold,
                                   max_job_length=window_size, 
                                   save_dir=args.process_high_denial_output,
                                   sort=True,
                                   col_name='denied_dateTime',
                                   new_col_name='denials_past_period',
                                   load_saved_file=args.process_high_denial_load
                                   )
        
    
    if args.process_data:
        file = args.process_file#"data/comsol2015.csv"
        if not os.path.isdir(args.process_dir):
            print(f"Making directory {args.process_dir}")
            os.makedirs(args.process_dir)
        all_features = get_features(file)
        print(all_features)
        data_list = []
        pass_list = [(file, os.path.join(args.process_dir, f"{f}{args.process_extension}"), f) for f in all_features]
        with mp.Pool(processes = mp.cpu_count()) as p:
            data_list = list(p.map(parallel_process, pass_list))
    if args.process_even_spacing:
        dir_name = os.path.dirname(os.path.abspath(args.even_spacing_output))
        if not os.path.isdir(dir_name):
            print(f"Making directory {dir_name}")
            os.makedirs(dir_name)
        #file = args.process_file 
        #all_features = list(get_features(file))
        from os import listdir
        from os.path import isfile, join
        all_features = [f.split('.')[0] for f in listdir(args.process_dir) if isfile(join(args.process_dir, f))]
        
        #all_features.remove('lum_fdtd_solve') # TODO why doesn't this file work????


        print(all_features)
        intervals = (args.S, args.M, args.H) # second, minute, hour
        pass_list = [(os.path.join(args.process_dir, f"{f}{args.process_extension}"), f, intervals) for f in all_features]
        with mp.Pool(processes = mp.cpu_count()-2) as p:
            data_list = list(p.map(parallel_even_spaceing, pass_list))
        #data_list = []
        #for p in pass_list:
        #    print(f"processing even spacing for file {p[1]}")
        #    data_list.append(parallel_even_spaceing(p))
        comb = pd.concat(data_list, axis=1).fillna(0)
        comb.to_csv(args.even_spacing_output)
        
    if args.process_license_duration:
        data_file = args.process_file
        data = pd.read_csv(data_file, index_col=0, parse_dates=[0])
        freq = args.process_license_duration_window_size
        window = pd.to_timedelta(freq if freq[0].isdecimal() else '1' + freq)
        print(f"window size = {window}")
        log_interval_file = args.process_license_duration_log_file
        data = generate_feature_avg_license_duration(data, window, log_interval_file, col_name="avg_duration", cpu_count=mp.cpu_count(), save_dir=args.process_license_duration_output)

    if args.to_long_format:
        # TODO do this without process file by reading dir file names
        file = args.process_file 
        all_features = get_features(file) # TODO get this information from files in directory
        print(all_features)
        file_list = [(os.path.join(args.process_dir, f"{f}{args.process_extension}"), f) for f in all_features]
        long_data = to_long_format(file_list)
        long_data.to_csv(args.long_format_output, index=False)
    #data = pd.read_csv("./processed/comsol2015/combined_1day_crop_COMSOL.csv", index_col=0, parse_dates=[0])
    #print(data.shape)
    
    #even_spacing(data=pd.read_csv('./processed/comsol2015/MEMSBATCH.csv', index_col=0, parse_dates=[0]), seconds=0, minutes=0, hours=1, col_name='MEMSBATCH')
