import os
import datetime
import bisect
import pandas as pd
import multiprocessing as mp
import tqdm
from tqdm.contrib.concurrent import process_map
import sys
import utils
from typing import List
import random
import argparse

 
def arg_parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--processed_license_log', type=str, default='', help='Location of the processed license log file')
    parser.add_argument('--accounting_log_dir', type=str, default='', help='Location of accounting logs')
    parser.add_argument('--output_file', type=str, default='', help='Where to save simulator data')
    parser.add_argument('--max_job_time', type=int, default=48+6, help='Maximum time of executing jobs in hours. Should add a small time delta to deal with logging delays.')
    parser.add_argument('--start_date', type=str, default='', help='Start date to generate data. Format: yyyy-mm-dd')
    parser.add_argument('--end_date', type=str, default='', help='End date to generate data. Format: yyyy-mm-dd')
    return parser.parse_args()

'''
    Generate file location of accounting log file
'''
def get_log_fname(license_date, log_directory, file_suffix): 
    log_date = license_date.strftime('%Y%m%d')   
    return os.path.join(log_directory, log_date+file_suffix)


'''
    Convert date and time in accounting log files to DateTime object
'''
def make_datetime(datetime_str):
    return datetime.datetime.strptime(datetime_str, '%m/%d/%Y %H:%M:%S')
    
    
'''
    Read accounting log file and return file as list.
    Ignores queue lines.
'''
def read_log(fname, version=0, data_dict={}, show_warnings=False):
    # check if file exists
    try: 
        open(fname, 'r')
    except:
        print(f'Failed to open {fname}')
        return []
    if version == 0:
        data = []
        with open(fname, 'r') as log_file:
            for ind, line in enumerate(log_file):
                _line = line.split(";")
                # get rid of queue lines, since they are not useful
                if _line[1] == 'S' or _line[1] == 'E':
                    _line[0] = make_datetime(_line[0])
                    _line[3] = {k:v for k,v in [i.split("=",1) for i in _line[3].split(" ")]}
                    _line.append(ind)
                    data.append(_line)
        return data
    else:
        with open(fname, 'r') as log_file:
            for ind, line in enumerate(log_file):
                #print(f"file:{log_file} line:{ind}")
                _line = line.split(";")
                job_id = _line[2].split('.')[0]
                date_time = make_datetime(_line[0])
                if data_dict.get(job_id) == None:
                    data_dict[job_id] = {'Q':[], 'S':None, 'E':None, 'D':[], 'A':[]} # Note: can have multiple Q for single job id, if moved to another queue
                
                if _line[1] == 'S' or _line[1] == 'E':
                    #_line[0] = make_datetime(_line[0])
                    #_line[3] = {k:v for k,v in [i.split("=",1) for i in _line[3].split(" ")]}
                    #_line.append(ind)
                    
                    key = _line[1]
                    
                    if data_dict[job_id][key] != None:
                        if show_warnings and key == 'S': # multiple S caused by error in compute node. Get last S time
                            print (f"WARNING, jobid: {job_id} already has entry {data_dict[job_id][key]} for {key} in file {fname} line {ind}")
                        if key == 'E': 
                            if show_warnings:
                                print(f"WARNING, jobid: {job_id} already has entry {data_dict[job_id][key]} for {key} in file {fname} line {ind}")
                            continue # Get first E time
                    
                    data_dict[job_id][key] = (date_time, {k:v.rstrip() for k,v in [i.split("=",1) for i in _line[3].split(" ")]})
                    
                elif _line[1] == 'Q':
                    queue_name = _line[3].split('=')[1]
                    data_dict[job_id]['Q'].append((date_time, queue_name.rstrip()))
                elif _line[1] == 'D':
                    key = _line[1]
                    data_dict[job_id][key].append(date_time)
                elif _line[1] == 'A':
                    key = _line[1]
                    data_dict[job_id][key].append(date_time)
                else:
                    #print(f"ignoring line: {line}")
                    pass
'''
    Checks if node_name is in exec_host in accounting logs
'''
def is_node(node, exec_host):
    # shorten node name, since full name is not listed in exec_host
    _node = node.split('.')[0]
    return _node in exec_host


'''
    some nodes are not logged, skip these nodes for speedup.
    Returns True if node name is not logged
'''
def skip_node(node):
    if "hammer" in node:
        return True
    elif "HAMMOND" in node:
        return True
    elif "ATHERTN" in node:
        return True
    elif "lgn" in node: # login nodes are not recorded in accounting logs
        return True
    return False


'''
    Search for job forward or backwards, used by find_job()
'''
def _search(license_date, user, node, direction="forward", max_exec_time=datetime.timedelta(hours=12), log_directory=None, file_suffix=None):
    if skip_node(node):
        return None
    
    log_fname = get_log_fname(license_date, log_directory, file_suffix)
    data = read_log(log_fname)
    
    # log_file doesn't exist; missing or no jobs logged that day
    # NOTE: Assuming that max_exec_time is less than 1 day
    if data == []:
        return None
    
    stop = False
    found_index = -1
    if direction == "backward":
        start_ind = bisect.bisect_right(data, license_date, key=lambda line: line[0])-1 # find rightmost index if duplicates exist
        stop_ind = -1
        step = -1
        action = 'S' # search for job start
        next_file_delta = -datetime.timedelta(days=1) # previous date
    elif direction == "forward":
        start_ind = bisect.bisect_left(data, license_date, key=lambda line: line[0]) # find rightmost index if duplicates exist
        stop_ind = len(data)
        step = 1
        action = 'E' # search for job end
        next_file_delta = datetime.timedelta(days=1) # next date
    else:
        raise ValueError(f"Invalid value {direction} for parameter 'direction'")
    
    next_file_ctr = 1
    while (not stop):
        for i in range(start_ind, stop_ind, step):
            #print(f"logfname {log_fname}, nextfilectr {next_file_ctr}, {direction}, index {i}, len data {len(data)}")
            cur_date = data[i][0]
            
            # limit how far forward/back we search
            if (abs(license_date-cur_date) > max_exec_time):
                stop = True
                break
            # check if we found job start
            # _data[i][3] is a dictionary
            if data[i][3].get("user") == user \
                and is_node(node, data[i][3].get("exec_host")) \
                    and data[i][1] == action:
                        stop = True
                        found_index = i
                        break
        if not stop and log_directory != None:
            # open next log file and continue searching
            next_log_fname = get_log_fname(license_date+(next_file_delta*next_file_ctr), log_directory, file_suffix)
            next_file_ctr += 1
            log_fname = next_log_fname
            data = read_log(next_log_fname)
            #print(f"next log fname {next_log_fname}")
            #print(f"license date {license_date},,  current_date {data[0][0]}")
            if data == []:
                stop = True
            else:
                if direction == "backward":
                    start_ind = bisect.bisect_right(data, license_date, key=lambda line: line[0])-1
                else: # direction == "forward"
                    start_ind = bisect.bisect_left(data, license_date, key=lambda line: line[0])
                    stop_ind = len(data)
                
    if found_index == -1: # could not match license to job
        return None
    else:
        assert (data != [])
        return (data[found_index])
    

'''
    Match a job to given license in function parameters.
    Return unique id for job.
    Returns None if could not match with job.
'''
def find_job_id(license_check_out, license_check_in, user, node, log_directory, file_suffix='_deidentified', max_exec_time=datetime.timedelta(hours=12)):
    #print(f"starting {license_check_out}")
    if skip_node(node):
        #print(f"{license_check_out}    SKIP")
        return "SKIP"
    
    license_check_out = datetime.datetime.strptime(license_check_out, '%Y-%m-%d %H:%M:%S')
    license_check_in = datetime.datetime.strptime(license_check_in, '%Y-%m-%d %H:%M:%S')
    
    job_start = _search(license_check_out, user, node, direction="backward", max_exec_time=max_exec_time, log_directory=log_directory, file_suffix=file_suffix)
    job_end  = _search(license_check_in, user, node, direction="forward", max_exec_time=max_exec_time, log_directory=log_directory, file_suffix=file_suffix)
    
    
    # generate unique job name using ind and date
    '''
    if job_start == None:
        job_start_id = "NOT_FOUND"
    else:
        ind_start = job_start[4]
        job_start_id = job_start[0].strftime('%Y%m%d') + "_" + str(ind_start)
    if job_end == None:
        job_end_id = "NOT_FOUND"
    else:
        ind_end = job_end[4]
        job_end_id = job_end[0].strftime('%Y%m%d') + "_" + str(ind_end)
    #print(f"ending {license_check_out}   {job_start_id}:{job_end_id}")
    
    return f"{job_start_id}:{job_end_id}"
    '''
    
    # job logs contain unique id for each job
    status = 0
    if job_start == None:
        job_start_id = "NOT_FOUND"
    else:
        job_start_id = job_start[2].split('.')[0]
        status += 1
    if job_end == None:
        job_end_id = "NOT_FOUND"
    else:
        job_end_id = job_end[2].split('.')[0]
        status += 1
        
    if status == 2:
        try:
            assert(job_start_id == job_end_id)
        except:
            print(f"WARNING: JOB MISMATCH, {job_start_id} and {job_end_id}")
            return(f"MISMATCH_{job_start_id}:{job_end_id}")
        return job_start_id
    elif status == 1:
        return(f"PARTIAL_{job_start_id}:{job_end_id}")
    else: # status == 0:
        return "NOT_FOUND"

def parallel_find_job_id(pass_list):
    check_out, check_in, user_name, node, log_directory, file_suffix, max_exec_time = pass_list
    return find_job_id(check_out, check_in, user_name, node, log_directory, file_suffix, max_exec_time)

def match_jobs(license_log_fname, log_directory, 
               output_file=None, file_suffix=None, 
               max_job_exec_time=datetime.timedelta(hours=12), 
               show_progress=True, n_cpu=16, 
               lines_start=None, lines_end=None, chunksize=32):

    
    license_log_file = open(license_log_fname, 'r')
    next(license_log_file) # skip header
    i = 0
    t = datetime.datetime.now()
           
    pass_list = [(*line.split(',')[2:6], log_directory, file_suffix, max_job_exec_time) for line in license_log_file]

    pass_list = pass_list[lines_start:lines_end]
        
    # remove any whitespace lines at end of file
    while len(pass_list[-1]) < 7:
        pass_list.pop()
        
    with mp.Pool(processes = n_cpu) as p:
        #job_list = list(tqdm.tqdm(p.imap(parallel_find_job_id, pass_list), total=len(pass_list)-1))
        print("Started processing")
        job_list = process_map(parallel_find_job_id, pass_list, max_workers=n_cpu, chunksize=chunksize)
        
    license_log_file.close()
    
    if output_file != None:
        #TODO: partial update output file without opening entire file
        if lines_start!=None or lines_end!=None:
            # try to open output file if it exists
            print("trying to open in incomplete output file")
            try:
                df = pd.read_csv(output_file, header=0, parse_dates=[2,3], dtype={'job_id' : str})
            except:
                print("Failed")
                df = pd.read_csv(license_log_fname, header=0, parse_dates=[2,3])
        else:
            df = pd.read_csv(license_log_fname, header=0, parse_dates=[2,3])    
                   
        # update log with job ids
        #df = pd.read_csv(license_log_fname, header=0, parse_dates=[2,3])
        
        if "job_id" not in list(df.columns):
            df['job_id'] = None
        #while(len(job_list) < len(df)):
        #    job_list.append(None)
        
        #df['job_id'][lines_start:lines_end] = job_list
        df.loc[lines_start:(lines_end-1), ('job_id')] = job_list

        df.to_csv(output_file, index=False)
        
    return job_list


def run_job_matching():
    if sys.version_info < (3, 10):
        print("Warning, Python 3.10 or greater required!")
    license_log_fname = "./data/comsol2015_sorted.csv"
    accounting_log_directory = "D:/logs/accounting"
    progress_file = "./data/partial/progress.txt"
    output_file = "./data/partial/comsol2015_jobs.csv"
    accounting_log_file_suffix = "_deidentified"
    max_job_exec_time = datetime.timedelta(hours=48) # smaller is faster but less accurate
    n_cpu = mp.cpu_count()
    #df = pd.read_csv(license_log_fname, header=0, parse_dates=[2,3]).sort_values('check_out', ascending=False)
    #df.to_csv("./data/comsol2015_sorted.csv", index=False)
    compute_step = 100#120922
    chunksize=min((compute_step//n_cpu)//4,2)
    print(f"n_cpu = {n_cpu}, chunksize={chunksize}")
    num_lines = sum(1 for _ in open(license_log_fname))-1 # -1 to skip header
    line_progress = 0
    
    # try resuming progress
    try:
        with open(progress_file, 'r') as progress:
            line_progress = int(progress.read())
            print(f"Resuming from line {line_progress}")
    except:
        pass
    
    for lines_start in range(line_progress, num_lines, compute_step):
        with open(progress_file, 'w') as progress:
            progress.write(str(lines_start))
        
        lines_end = lines_start+compute_step
        t = datetime.datetime.now()
        match_jobs(license_log_fname, accounting_log_directory, 
                output_file=output_file, file_suffix=accounting_log_file_suffix, 
                max_job_exec_time=max_job_exec_time, n_cpu=n_cpu, 
                lines_start=lines_start, lines_end=lines_end, chunksize=chunksize)
        print(f"total time for lines {lines_start}:{lines_end} {datetime.datetime.now()-t}")    
 

###############
'''
Match job Queue, Start and End with matching job ids in a single file
'''
def match_job_intervals(license_log_fname, log_directory: str, 
                        output_file=None, file_suffix=None, 
                        start_date: datetime.datetime = None, 
                        end_date: datetime.datetime = None, 
                        show_progress=True, show_warnings = False, n_cpu=16):
    
    # TODO give this function as input parameter
    def fileNameToDate(fname):
        datetime_str = fname.split('_')[0]
        return datetime.datetime.strptime(datetime_str, '%Y%m%d')
    
    data = {}
    log_flist=os.listdir(log_directory)
    
    for fname in reversed(log_flist):
        fname_date = fileNameToDate(fname)
        if fname_date < start_date or fname_date > end_date:
            log_flist.remove(fname)
    
    #get_log_fname(license_date, log_directory, file_suffix)
    #make_datetime(datetime_str)
        
    for ind, log_fname in enumerate(log_flist):
        f = os.path.join(log_directory, log_fname)
        if show_progress:
            print(f"Opened file {f}  {ind+1}/{len(log_flist)}")
        read_log(f, version=1, data_dict=data, show_warnings=show_warnings)
        
    return data

def _parallel_match_license_to_job(pass_list):
    job_data, max_job_exec_time, start_sorted, end_sorted, optimize_query, check_in_buffer_time, _ = pass_list
    job_data: pd.DataFrame
    license_name, license_checkout, license_checkin, user_name, node = _
    
    #padding = datetime.timedelta
    
    ## do the following operations to decrease query space search size TODO unfinished
    if optimize_query:
        start_left = start_sorted.searchsorted(license_checkout - (max_job_exec_time), side='left')
        start_right = start_sorted.searchsorted(license_checkout - (max_job_exec_time), side='right')
        start_ind = end_sorted.index[start_left:start_right+1].min()
        
        #print(f"start left, start right = {start_left},{start_right}")
        #print(f"{type(end_sorted)}")
        #print(f"{end_sorted.index[start_left:start_right+1]}")
        
        end_left = end_sorted.searchsorted(license_checkin + (max_job_exec_time), side='left')
        end_right = end_sorted.searchsorted(license_checkin + (max_job_exec_time), side='right')
        #if end_right == len(end_sorted):
        #    end_right -= 1

        if end_left == len(job_data):
            assert(end_left<=end_right)
            end_ind = len(job_data)
        else:
            end_ind = end_sorted.index[end_left:end_right+1].max()
        
        if type(start_ind) == float or type(end_ind) == float:
            print(f"end left={end_left}  end_right={end_right}   end inds = {end_sorted.index[end_left:end_right+1]}")
            raise Exception(f"start ind={start_ind}, end ind={end_ind}  type startind = {type(start_ind)}, type ending = {type(end_ind)}")
        

        #matches = job_data.query(f'(@job_start_min <= `job_start` <= @license_checkout) and (@license_checkin <= `job_end` <= @job_end_max) and `user_name` == @user_name and `node`.str.contains(@node)')

        job_data_ = job_data.iloc[start_ind:end_ind+1]
        #print(f"size of job data search space decreased by {len(job_data)-len(job_data_)}")
    else:
        job_data_ = job_data
        
    job_start_min = license_checkout - max_job_exec_time
    job_end_max = license_checkin + max_job_exec_time + check_in_buffer_time
    
    job_data_ = job_data_.query(f'@job_start_min <= `job_start` and `job_end` <= @job_end_max') # is this required?
    license_checkin_ = license_checkin - check_in_buffer_time
    matches = job_data_.query(f'(`job_start` <= @license_checkout < `job_end` and `job_end` >= @license_checkin_) and `user_name` == @user_name and `node`.str.contains(@node)')
    return matches

def match_license_to_job(license_log_fname: str, job_interval_data: dict, output_file: str, max_job_exec_time: datetime.timedelta, show_progress=False, n_cpu=8, optimize_query=True):
    
    # convert jobs into a better format
    columns = ['job_id', 'job_start', 'job_end', 'job_queue', 'queue_name', 'job_name', 'user_name', 'node', 'time_in_queue']
    data = []
    
    for job_id, value in job_interval_data.items():
        assert(type(value['Q'])==list)
        if len(value['Q']) == 0 or value['S'] == None:
            #print(f"skipping job {job_id}")
            continue
        substitute_D = False
        substitute_A = False

        if value['E'] == None: 
            if len(value['D']) == 1: # sometimes when job gets terminated (D) there is no E. Also some jobs continuously get D, also do not want those
                print(f"substituting S for D in job {job_id}")
                substitute_D = True
            elif len(value['A']) == 1: # sometimes when job gets aborted? (A) there is no E.
                print(f"substituting S for A in job {job_id}")
                substitute_A = True
            else:
                continue

        job_queue, queue_name = value['Q'][0]
        job_start = value['S'][0]
        if substitute_D:
            job_end = value['D'][0]
        elif substitute_A:
            job_end = value['A'][0]
        else:
            job_end = value['E'][0]
        job_name = value['S'][1].get('jobname', 'None')
        user_name = value['S'][1].get('user')
        node = value['S'][1].get('exec_host')#.split('/')[0]
        time_in_queue = job_start - job_queue
        if job_end == None or job_start == None or job_queue == None or user_name == None:
            raise Exception
        data.append([job_id, job_start, job_end, job_queue, queue_name, job_name, user_name, node, time_in_queue])
    
    print(f"Number of jobs intervals inside time period {len(data)}")

    job_data = pd.DataFrame(data, columns=columns)
    start_sorted = job_data.sort_values(by=['job_start'])['job_start']
    end_sorted = job_data.sort_values(by=['job_end'])['job_end']
    min_job_start = job_data['job_start'].min()
    max_job_end = job_data['job_end'].max()
    check_in_buffer_time = datetime.timedelta(minutes=1) # there may be some delay between check in and job end, leading to check in time being a few seconds after job end
    
    max_job_end_plus_buff = max_job_end+check_in_buffer_time
    
    print(f"min job start: {min_job_start}, max job end: {max_job_end_plus_buff}")
    #print(job_data)
    license_data = pd.read_csv(license_log_fname, header=0, parse_dates=[2,3])
    
    # remove 0 duration license intervals
    license_data = license_data[license_data['delta'] != 0]

    # want all license intervals that fit exactly inside the time period, plus buffer
    license_data = license_data.query(f'(@min_job_start <= `check_out` < @max_job_end) and (@min_job_start <  `check_in` <= @max_job_end_plus_buff)')
    #license_data = license_data.query(f'(@min_job_start <= `check_out` < @max_job_end) or (@min_job_start <  `check_in` <= @max_job_end)')
    
    license_count = len(license_data)
    
    pass_list = [(job_data, max_job_exec_time, start_sorted, end_sorted, optimize_query, check_in_buffer_time, x) for x in zip(license_data['feature_names'], license_data['check_out'], license_data['check_in'], license_data['user_name'], license_data['node'])]
    
    print("Started processing")
    print(f"total Licenses to match: {len(license_data)}")
    
    chunksize=max((license_count//n_cpu)//16,16)
    print(f"n_cpu = {n_cpu}, chunksize={chunksize}")
    
    matches = process_map(_parallel_match_license_to_job, pass_list, max_workers=n_cpu, chunksize=chunksize)
    matches: List[pd.DataFrame]
    print("done")
    #matches = [x for x in matches_ if x != []]
    #print(len(matches))
    '''
    for license_name, license_checkout, license_checkin, user_name, node in zip(log_data['feature_names'], log_data['check_out'], log_data['check_in'], log_data['user_name'], log_data['node']):#log_data[['check_out', 'check_in']]:
        if show_progress:
            i+=1
            
            print(f"Progress: {i}/{log_count} licenses")
        assert(type(value['Q'])==list)
        # NOTE there may be multiple node names in the value of exec_nodes, so we search for a single match in that string 
        matches = job_data.query(f'(`job_start` <= @license_checkout and `job_end` >= @license_checkin) and `user_name` == @user_name and `node`.str.contains(@node)')
        #exec_host=comp-sc-0512/0-7+comp-sc-0509/0-7+comp-sc-0507/0-7+comp-sc-0503/0-7+comp-sc-0501/0-7+comp-sc-0500/0-7+comp-sc-0499/0-7+comp-sc-0498/0-7+comp-sc-0497/0-7+comp-sc-0496/0-7+comp-sc-0494/0-7+comp-sc-0493/0-7+comp-sc-0492/0-7+comp-sc-0490/0-7+comp-sc-0489/0-7+comp-sc-0488/0-7+comp-sc-0478/0-7+comp-sc-0477/0-7+comp-sc-0474/0-7+comp-sc-0472/0-7 Resource_List.nodes=20:ppn=8 Resource_List.walltime=48:00:00 Resource_List.pmem=24gb Resource_List.nodect=20 Resource_List.neednodes=20:ppn=8
        if len(matches) == 0:
            unmatched_licenses.append((license_name, license_checkout, license_checkin, user_name, node))
            #print(f"license_name={license_name}, license_checkout={license_checkout}, license_checkin={license_checkin}, user_name={user_name}, node={node}")
    
    '''

    # matches hold what jobs a license may belong to
    
    # license _data columns: feature_names,delta,check_out,check_in,user_name,node
    # job_data columns: ['job_id', 'job_start', 'job_end', 'job_queue', 'queue_name', 'job_name', 'user_name', 'node']
    
    licenses_of_job = {} # key:jobid, value = [licenses]
    unmatched_licenses = []
    randomly_chosen_job_count_list = []
    for ind, license_ in enumerate(license_data.iterrows()):
        _, license = license_
        license: pd.Series
        #print('======')
        #print(f"ind {ind}  license {license}")
        jobs = matches[ind]
        if len(jobs) == 0:
            unmatched_licenses.append(license)
            continue
        elif len(jobs) == 1:
            j = jobs.iloc[0]
            job_id = j['job_id']
            if licenses_of_job.get(job_id) == None:
                licenses_of_job[job_id] = []
            assert(j['user_name'] == license['user_name'])
            if j['job_end'] < license['check_in']:
                print(f"WARNING: job {job_id} end time={j['job_end']},  license {license['feature_names']} check_in = {license['check_in']} ")
            licenses_of_job[job_id].append(license)
        else: # multiple job matches
            print('======')
            print(f"WARNING: license {license['feature_names']} matching jobs:")
            randomly_chosen_job_count_list.append(len(jobs))
            candidate_jobs = [] # keeps jobs that do not require check in buffer time added to job end
            all_jobs = []
            for _, j in jobs.iterrows():
                all_jobs.append(j)
                print(f"\tjob {j['job_id']} {j['job_name']}, {j['job_start']} - {j['job_end']}")
                # check if any 
                if not (j['job_end'] < license['check_in']): # check if any possibility without check in buffer time added
                    candidate_jobs.append(j)
            if len(candidate_jobs) > 0:
                if len(candidate_jobs) == 0:
                    j_id = candidate_jobs[0]['job_id']
                    if licenses_of_job.get(j_id) == None:
                        licenses_of_job[j_id] = []
                    licenses_of_job[j_id].append(j_id)
                else:
                    # choose random job
                    j_id = random.choice(candidate_jobs)['job_id']
                    if licenses_of_job.get(j_id) == None:
                        licenses_of_job[j_id] = []
                    licenses_of_job[j_id].append(license)
            else:
                # choose random job if no candidate job
                j_id = random.choice(all_jobs)['job_id']
                if licenses_of_job.get(j_id) == None:
                    licenses_of_job[j_id] = []
                licenses_of_job[j_id].append(license)

    # convert to simulator input format and save to file
    
    job_data.sort_values(by=['job_queue'])
    text = ''
    for _, job in job_data.iterrows():
        # Format of each line in file: jobId, queueTime, jobDuration, queueName; licenseName, checkOut(time in sec after job starts), licenseDuration; licenseName, checkOut, licenseDuration; ...
        
        job_id = job['job_id']
        queue_time = job['job_queue'].strftime('%Y-%m-%d %H:%M:%S') # aka submission time
        job_duration = int(datetime.timedelta.total_seconds((job['job_end'] - job['job_start'])))
        if job_duration == 0:
            continue
        queue_name = job['queue_name']
        time_in_queue = int(datetime.timedelta.total_seconds(job['time_in_queue']))
        text += f"{job_id},{queue_time},{job_duration},{queue_name},{time_in_queue}"
        
        if licenses_of_job.get(job_id) != None:
            license_list: List[pd.Series] = licenses_of_job.get(job_id)
            for license in license_list:
                license: pd.Series
                license_name = license['feature_names']
            
                license_checkout_time = int((license['check_out'] - job['job_start']).total_seconds())
                #license_checkout_time = int(datetime.timedelta.total_seconds(license_checkout_time))
                assert(license_checkout_time >= 0)
                
                license_checkin = min(license['check_in'], job['job_end']) # there may be a few seconds difference in log time if license checked out and job ended at same 
                license_duration = int((license_checkin - license['check_out']).total_seconds())
                assert(license_checkout_time+license_duration<=job_duration)
                assert(license_duration>0)
                text += f";{license_name},{license_checkout_time},{license_duration}"
        text += '\n'
    text = text.rstrip()
    
    with open(output_file, 'w') as f:
        f.write(text)   
    
    # Get statistics about this job interval
    
    job_time_to_complete_time_avg = job_data.apply(lambda row : (row['job_end']-row['job_queue']).total_seconds(), axis=1).mean(axis=0)
    job_queue_time_avg = job_data.apply(lambda row : (row['job_start']-row['job_queue']).total_seconds(), axis=1).mean(axis=0)
    job_queue_time_median = job_data.apply(lambda row : (row['job_start']-row['job_queue']).total_seconds(), axis=1).median(axis=0)       
    return unmatched_licenses, licenses_of_job, (randomly_chosen_job_count_list, job_time_to_complete_time_avg, job_queue_time_avg, job_queue_time_median)


 # for debugging
def convert(job_interval_data):
    
    # convert jobs into a better format
    columns = ['job_id', 'job_start', 'job_end', 'job_queue', 'queue_name', 'job_name', 'user_name', 'node']
    data = []
    for job_id, value in job_interval_data.items():
        assert(type(value['Q'])==list)
        if len(value['Q']) == 0 or value['S'] == None or value['E'] == None:
            #print(f"skipping job {job_id}")
            continue
        job_queue, queue_name = value['Q'][0]
        job_start = value['S'][0]
        job_end = value['E'][0]
        job_name = value['S'][1].get('jobname', 'None')
        user_name = value['S'][1].get('user')
        node = value['S'][1].get('exec_host').split('/')[0]
        if job_end == None or job_start == None or job_queue == None or user_name == None:
            raise Exception
        data.append([job_id, job_start, job_end, job_queue, queue_name,job_name, user_name, node])
    
    print(len(data))

    return pd.DataFrame(data, columns=columns)

#%% 
def get_cpu_requests(job_interval_data: dict):  
     
    cpu_usage_list = []
    for job_id, value in job_interval_data.items():
        
        # skip jobs similar to as done in function match_license_to_job()
        if len(value['Q']) == 0 or value['S'] == None:
            continue
        if value['E'] == None: 
            if len(value['D']) == 1: # sometimes when job gets terminated (D) there is no E. Also some jobs continuously get D, also do not want those
                pass
            else:
                continue

    
        
        #print(f"job {job_id} ", end='')
        if value['S'][1].get('Resource_List.nodes') != None:
    
            #print(f"{value['S'][1]['Resource_List.nodes']}")
            value_ = value['S'][1]['Resource_List.nodes'].split(':stmem')[0]
            cpu_usage = None
            if len(value_.split(':ppn=')) == 2:
                nodes, ppn = value_.split(':ppn=') # X:ppn=Y format where X and Y are integers
                if 'himem' in nodes:
                    nodes = nodes.split(':himem')[0]
                try:
                    nodes = int(nodes)
                except:
                    print(value_)
                    nodes = 1 # requested specific CPU
                ppn = int(ppn.split(':')[0])
                cpu_usage = (nodes)*(ppn)
            elif len(value_.split(':ppn=')) == 1:
                cpu_usage = value_.split(':ppn=')[0]
                if 'gpu' in cpu_usage:
                    cpu_usage = int(cpu_usage.split(':')[0])
                else:
                    cpu_usage = int(cpu_usage)
            elif '+' in value_:
                reqs = value_.split('+')
                cpu_usage = 0
                for req in reqs:
                    nodes, ppn = req.split(':ppn=')
                    ppn = int(ppn.split(':')[0])
                    try:
                        nodes = int(nodes)
                    except:
                        nodes = 1
                    cpu_usage += nodes*ppn
                assert(cpu_usage>1)

            else:
                print(value_)
                assert(False)

            cpu_usage_list.append(cpu_usage)
        elif value['S'][1].get('Resource_List.procs') != None:
            #print(f"{value['S'][1]['Resource_List.procs']}")
            procs = value['S'][1]['Resource_List.procs']
            procs = int(procs)
            cpu_usage_list.append(cpu_usage)
        else:
            pass
            #print(' SKIPPED')
            
    return cpu_usage_list

#%%
def generate_jobs(unmatched_licenses: list, output_file: str, max_job_exec_time: datetime.timedelta, tmp_result_file: str = None, queue_duration: datetime.timedelta = None):
    '''
    Turn unmatched licenses into jobs. Append result to output_file.
    '''
    
    data = pd.DataFrame(unmatched_licenses).reset_index().drop(columns=['index'])
    user_node_pairs = data.groupby(['user_name', 'node']).count().reset_index()
    to_tmp_file = ""
    to_output_file = ""
    job_id_num = 1
    # TODO make this faster
    for row in user_node_pairs.values:
        user_name, node, count, _, _, _ = row   
        matching_licenses = data.query(f'(`user_name` == @user_name) and (`node` == @node)')
        # Group licenses with regards to temporal locality
        clusters = []
        chosen_indices = []
        for l in matching_licenses.values:
            _license_name, _delta, check_out, check_in, _user_name, _node = l
            #check_in = l["check_out"]
            #check_out = l["check_in"]
            cluster = matching_licenses.loc[~matching_licenses.index.isin(chosen_indices)].query(f'`check_out` <= @check_in')
            if len(cluster) > 0 and check_out >= cluster["check_in"].max():
                clusters.append(cluster)
                chosen_indices += list(cluster.index)
        
        
        print(clusters)
        assert(len(clusters)>0)
        for cluster in clusters:
            job_id = f"gen_{job_id_num}"
            job_id_num += 1
            queue_name = "open"
            queue_duration = int(job_queue_time_avg)
            
            
            min_check_out: datetime.datetime = cluster["check_out"].min()
            max_check_in: datetime.datetime = cluster["check_in"].max()
            job_duration = int((max_check_in - min_check_out).total_seconds())
            if (job_duration > max_job_exec_time.total_seconds()):
                print(f"WARNING: Job length {job_duration}s for job_id:{job_id} user:{user_name} node:{node}")
                #assert(False)
                
            queue_time = (min_check_out-datetime.timedelta(seconds=queue_duration)).strftime('%Y-%m-%d %H:%M:%S') # aka submission time

            job_txt = ""
            for license in cluster.values:
                to_tmp_file += f"{','.join(str(x) for x in license)}\n"
                index, license_name, delta, check_out, check_in, _user_name, _node = license
                job_txt += f";{license_name},{(check_out-min_check_out).total_seconds()},{int(delta)}"
                
            to_tmp_file += "===========\n"
            to_output_file += f"{job_id},{queue_time},{job_duration},{queue_name},{queue_duration}{job_txt}\n"
    # remove last newline
    to_output_file = to_output_file[:-1]
    
    if tmp_result_file != None:
        with open(tmp_result_file, 'w') as f:
            f.write(to_tmp_file)
    if output_file != None:
        with open(output_file, 'a') as f:
            f.write(to_output_file)
        
#%%
if __name__ == '__main__':
     
    #run_job_matching()
    
    args = arg_parse()  
    
    arg_start_date = args.start_date #"2017-10-01"
    arg_end_date = args.end_date #"2017-11-03"
    license_log_fname = args.processed_license_log #"./data/comsol2015_old_method.csv"
    accounting_log_directory = args.accounting_log_dir #"D:/logs/accounting"
    start_date = utils.parse_date(arg_start_date) 
    end_date = utils.parse_date(arg_end_date)
    output_file = args.output_file #f"./processed/simulator/comsol2015_jobs_{arg_start_date}-{arg_end_date}.csv"
    max_job_exec_time = args.max_job_time #datetime.timedelta(hours=48) + datetime.timedelta(hours=6)

    #%%
    # get all valid job intervals (i.e jobs that start and end within this time period)
    job_intervals = match_job_intervals(license_log_fname, accounting_log_directory, start_date=start_date, end_date=end_date)
    #%%
    cpu_request_list = get_cpu_requests(job_intervals)
    print(f"Average cpu requests per job in interval {sum(cpu_request_list)/len(cpu_request_list)}")
    #df = convert(job_intervals)
    #%%
    n_cpu=16
    print(len(job_intervals.keys()))
    unmatched_licenses, licenses_of_job, stats = match_license_to_job(license_log_fname, job_intervals, output_file, max_job_exec_time, show_progress=True, n_cpu=n_cpu, optimize_query=False)
    randomly_chosen_job_count_list, job_time_to_complete_time_avg, job_queue_time_avg, job_queue_time_median= stats
    #%%
    print(f"unmatched licenses {len(unmatched_licenses)}")
    print(f"jobs with licenses {len(licenses_of_job)}")
    l_ctr = 0
    for job_id, licenses in licenses_of_job.items():
        '''
        print(f"job {job_id} has licenses: ", end='')
        for l in licenses:
            print(f"{l['feature_names']}, ", end='')
        print()
        '''
        l_ctr += len(licenses)
    print(f"number of matched licenses {l_ctr}")
    
    print(f"total random choice when matching licenses to jobs = {len(randomly_chosen_job_count_list)}")
    print(f"percentage of license jobs with ambiguity %{100*len(randomly_chosen_job_count_list)/l_ctr}")
    avg_rand_jobs = 0
    if len(randomly_chosen_job_count_list) != 0:
        avg_rand_jobs = sum(randomly_chosen_job_count_list)/len(randomly_chosen_job_count_list)
    print(f"average random choice jobs count = {avg_rand_jobs}")
    # %%
    print("job interval stats")
    print(f"Average job time to complete {job_time_to_complete_time_avg}")
    print(f"Average job queue time {job_queue_time_avg}")
    print(f"Median job queue time {job_queue_time_median}")
    # %%
    tmp_result_location = f"./tmp/unmatched_licenses_{arg_start_date}-{arg_end_date}"
    with open(tmp_result_location, 'w') as f:
        for l in unmatched_licenses:
            f.write(f"{str(l)}\n")
    # %%
    # Deal with unmatched licenses
    print("Dealing with unmatched licenses")
    #generate_jobs(unmatched_licenses, output_file, max_job_exec_time, tmp_result_file=tmp_result_location, queue_duration=job_queue_time_avg)
    # %%  
    data = pd.DataFrame(unmatched_licenses).reset_index().drop(columns=['index'])
    user_node_pairs = data.groupby(['user_name', 'node']).count().reset_index()
    to_tmp_file = ""
    to_output_file = "\n"
    job_id_num = 1
    split_license_name = "COMSOL" # name of license to split jobs, e.g. base license requried for other sub-licenses
    # TODO make this faster
    #%%
    for row in user_node_pairs.values:
        user_name, node, count, _, _, _ = row   
        matching_licenses = data.query(f'(`user_name` == @user_name) and (`node` == @node)')
        # Group licenses with regards to temporal locality
        clusters = []
        chosen_indices = []
        for l in matching_licenses.values:
            _license_name, _delta, check_out, check_in, _user_name, _node = l
            #check_in = l["check_out"]
            #check_out = l["check_in"]
            cluster = matching_licenses.loc[~matching_licenses.index.isin(chosen_indices)].query(f'`check_out` <= @check_in').sort_values(by=['check_out'])
            print(f"=====\n{cluster}\n----- {check_in} >= {cluster['check_in'].max()}")
            if len(cluster) > 0 and check_in >= cluster["check_in"].max():
                clusters.append(cluster)
                chosen_indices += list(cluster.index)
        

        print(clusters)
        assert(len(clusters)>0)
        clen = 0
        for c in clusters:
            clen += len(c)
        assert(clen == len(matching_licenses))
        
        
        
        # split across base license
        sub_clusters = []
        
        for cluster in clusters:
            sub_clusters_size = 0
            split_licenses = cluster.query(f'`feature_names` == @split_license_name').sort_values(by=['check_out'])
            if len(split_licenses) <= 1:
                assert(len(cluster)>0)
                sub_clusters.append(cluster)
                continue
                        
            for ind in range(len(split_licenses)):
                split = list(split_licenses.values)[ind]
                _license_name, _delta, check_out, check_in, _user_name, _node = split
                if ind+1 < len(split_licenses):
                    _, _, next_check_out, _next_check_in, _, _ = list(split_licenses.values)[ind+1]
                else:
                    next_check_out = pd.Timestamp.max
                if ind == 0:
                    sub_cluster = cluster.query(f'(`check_out` < @next_check_out)').sort_values(by=['check_out'])
                else:
                    sub_cluster = cluster.query(f'(`check_out` < @next_check_out) and (`check_out` >= @check_out)').sort_values(by=['check_out'])

                sub_clusters_size += len(sub_cluster)
                if not len(sub_cluster)>0:
                    print(f"WARNING: subcluster has length {len(sub_cluster)}")
                    print(f"num spliits = {len(split_licenses)}")
                    print(cluster)
                    print('=====')
                    print(sub_cluster)
                else:
                    sub_clusters.append(sub_cluster)
            assert(sub_clusters_size == len(cluster))
        
        



        for cluster in sub_clusters:
            job_id = f"gen_{job_id_num}"
            job_id_num += 1
            queue_name = "open"
            queue_duration = int(job_queue_time_avg)
            
            
            min_check_out: datetime.datetime = cluster["check_out"].min()
            max_check_in: datetime.datetime = cluster["check_in"].max()


            first_check_out_buffer = datetime.timedelta(seconds=40)
            job_start_time = min_check_out - first_check_out_buffer
            job_duration = int((max_check_in - job_start_time).total_seconds())
            if (job_duration > max_job_exec_time.total_seconds()):
                print(f"WARNING: Job length {job_duration}s for job_id:{job_id} user:{user_name} node:{node}")
                #assert(False)
                
            queue_time = (job_start_time-datetime.timedelta(seconds=queue_duration)).strftime('%Y-%m-%d %H:%M:%S') # aka submission time

            job_txt = ""
            for license in cluster.values:
                to_tmp_file += f"{','.join(str(x) for x in license)}\n"
                license_name, delta, check_out, check_in, _user_name, _node = license
                #license_duration = (check_out-min_check_out).total_seconds()
                if delta > 0.0:
                    job_txt += f";{license_name},{int((check_out-job_start_time).total_seconds())},{int(delta)}"
                
            to_tmp_file += "===========\n"
            to_output_file += f"{job_id},{queue_time},{job_duration},{queue_name},{queue_duration}{job_txt}\n"
    # remove last newline
    to_output_file = to_output_file[:-1]
    #%%
    print("writing to file")
    tmp_result_file = tmp_result_location
    if tmp_result_file != None:
        with open(tmp_result_file, 'w') as f:
            f.write(to_tmp_file)
    if output_file != None:
        with open(output_file, 'a') as f:
            f.write(to_output_file)  
# %%
