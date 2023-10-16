import argparse
import log_parse
import process
import multiprocessing as mp
import pandas as pd

import random
import os
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
    parser.add_argument('--log_file', type=str, default=None, help='File containing license log data')
    parser.add_argument('--start', type=str, default=None, help='Start Date of data to show statistics. Format: %Y-%m-%d Default starts at beginning')
    parser.add_argument('--end', type=str, default=None, help='Finish Date of data to show statistics, inclusive. Default finishes at end')
    
    parser.add_argument('--name', type=str, default=None, help='Unique name to use for generating files and data.')
    #parser.add_argument('--freq', type=str, default="D", help='Frequency of data set. Ex: 0D, H, min. For more information see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases')
    parser.add_argument('--fast_parse', type=str2bool, default=True, help="To use fast or slower method of log parsing")
    
    parser.add_argument('--max_license_file', type=str, default=None, help='Maximum license count file. Output of lmstat')
    parser.add_argument('--max_license_file_type', type=str, default='comsol', help='Type of max_license_file. Different licenses may have different formats')
    parser.add_argument('--skipped_duration', type=int, default=0, help='Length of time period to ignore in statistics calculations. Format: seconds')
    parser.add_argument('--show_statistics', type=str2bool, default=True, help='Whether to show various license statistics')


    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parse()    

    #data_file = args.data_file
    #data = pd.read_csv(data_file, index_col=0, parse_dates=[0])
    #data = utils.crop_data(data, args.start, args.end)    
    
    ##################
    ### parse logs ###
    ##################
    
    start_date = None
    if args.start is not None:
        start_date = utils.parse_date(args.start).date()
    end_date = None
    if args.end is not None:
        end_date = utils.parse_date(args.end).date()
    print(type(start_date))
    
    log_name = args.log_file   
    if args.name is None:
        name = random.randint(0,99999)
    else:
        name = args.name
        name.replace(" ", "_")
        
    my_license = log_parse.License(log_name, name, ignore_pid=False, ignore_numLicense=False, start_date=start_date, end_date=end_date)
    my_license.parse_file(parse_types = [])
    
    if args.fast_parse:
        skipped_keys = my_license.parse_pairs_old(maxPass=100)
    else:
        import datetime
        skipped_keys = my_license.parse_pairs(split_distance=datetime.timedelta(days=7), includeAllIntervals=False)
    my_license._parse_remaining_pairs(skipped_keys)
    
    license_interval_file = f"./data/{name}.csv"
    denied_file = f"./data/{name}_denied.csv"
    my_license.save_to_file(license_interval_file, 'pair')
    my_license.save_to_file(denied_file, 'denied')
    
    #################################
    ### process log interval data ###
    #################################
    
    process_dir = f"./processed/{name}_{random.randint(0,99999)}/"
    if not os.path.isdir(process_dir):
        print(f"Making directory {process_dir}")
        os.makedirs(process_dir)
    all_features = process.get_features(license_interval_file)
    print(all_features)
    data_list = []
    process_extension = '.csv'
    pass_list = [(license_interval_file, os.path.join(process_dir, f"{f}{process_extension}"), f) for f in all_features]
    with mp.Pool(processes = mp.cpu_count()) as p:
        data_list = list(p.map(process.parallel_process, pass_list))
    
    ############################    
    ### process even spacing ###
    ############################
    
    from os import listdir
    from os.path import isfile, join
    all_features = [f.split('.')[0] for f in listdir(process_dir) if isfile(join(process_dir, f))]

    print(all_features)
    intervals = (0, 0, 24) # second, minute, hour
    pass_list = [(os.path.join(process_dir, f"{f}{process_extension}"), f, intervals) for f in all_features]
    with mp.Pool(processes = mp.cpu_count()-2) as p:
        data_list = list(p.map(process.parallel_even_spaceing, pass_list))

    forecast_data = pd.concat(data_list, axis=1).fillna(0)
    even_spacing_dir = f"./processed/combined_{name}"
    if not os.path.isdir(even_spacing_dir):
        print(f"Making directory {even_spacing_dir}")
        os.makedirs(even_spacing_dir)
    even_spacing_output = os.path.join(even_spacing_dir, "combined_1day.csv")
    forecast_data.to_csv(even_spacing_output)
    
    ######################
    ### get statistics ###
    ######################    
    if args.show_statistics:
        duration_file = license_interval_file
        forecast_file = even_spacing_output
        max_license_file = args.max_license_file
        denial_file = denied_file
        
        max_licenses = None
        if max_license_file != None:
            import utils
            max_license_file_type = args.max_license_file_type 
            max_licenses = utils.parse_max_licenses(max_license_file, license_type=max_license_file_type, multiple_license_versions_behavior='max_version') #TODO auto type detect
            print(max_licenses)
        skipped_duration =  args.skipped_duration
        process.get_license_usage_statistics(duration_file, forecast_file, denial_file, max_licenses=max_licenses, skipped_time=skipped_duration)
        
    ############################
    ### Removing Directories ###
    ############################
    print(f"removing dir: {process_dir}")
    os.remove(process_dir)