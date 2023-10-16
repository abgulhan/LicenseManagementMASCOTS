import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import plotly.express as px
import os

monthToInt = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
global_date = 0 

class License:
    def __init__(self, log_name, name, ignore_pid = False, ignore_numLicense = False, start_date = None, end_date = None):
        self._log_name = log_name
        self._date = None
        #self._checkOuts = defaultdict(list)
        #self._checkIns = defaultdict(list)
        self._licenseChecks = defaultdict(list) # hash table, 
                                                # key=unique_identifier = (feature name, username + node, process id), 
                                                # value = (type, dateTime, number of licenses checked out)
        self._intervals = defaultdict(list) # key = feature name, value = list of delta time, check_in time, check_out time, username, node
        self._checkOuts = dict() # key - feature name, value = number of currently checked out licenses
        self._lineGraphData = pd.DataFrame(columns=["dateTime"]) # first column = datetime, other columns = number of checkouts for each feature
        ###self._lineGraphData.set_index("dateTime", inplace=True)
        self._name = name
        self._prev_line_time = datetime.time.min
        self._ignore_pid = ignore_pid
        self._ignore_numLicense = ignore_numLicense
        self._denied = defaultdict(list) # dictionary, key:feature name, value:(dateTime)
        self._restarts = [] # List of datetime.datetime
        self._start_date: pd.Timestamp = start_date
        self._end_date: pd.Timestamp = end_date
    
    '''
    Add data to some data structures.
    parse_types is used to ignore some data structures for faster processing
    '''
    def _add_data(self, type, data, parse_types = ["linegraph"]):
        date = data[0].date()
        if self._start_date is not None and date < self._start_date:
            return
        if self._end_date is not None and date > self._end_date:
            return
        if type == "OUT" or type == "IN":
            dateTime, uniqueId = data
            feature, user, pid, numLicense = uniqueId
            #assert('@' in user)
            #user_name, node = user.split('@')
            self._licenseChecks[uniqueId].append((type, dateTime))

            if "linegraph" in parse_types:
                if feature not in self._checkOuts:
                    self._checkOuts[feature] = 0
                    self._lineGraphData[feature] = 0
                if type == "OUT":
                    self._checkOuts[feature] += numLicense
                else: # type == "IN"
                    self._checkOuts[feature] -= numLicense
                new_row = self._checkOuts.copy()
                new_row["dateTime"] = pd.to_datetime(dateTime)#dateTime.timestamp() 
                ###self._lineGraphData = pd.concat([self._lineGraphData, pd.DataFrame([new_row]).set_index("dateTime")], axis=0, ignore_index=False)
                # TODO THIS IS VERY SLOW FIX THIS
                self._lineGraphData = pd.concat([self._lineGraphData, pd.DataFrame([new_row])], axis=0, ignore_index=True)
                #self._lineGraphData.append(new_row)
        elif type == "DENIED":
            dateTime, feature, user = data
            assert('@' in user)
            user_name, node = user.split('@')
            node = node.split('.')[0] # remove suffix. since it
            if feature not in self._denied:
                self._denied[feature] = []
            self._denied[feature].append((dateTime, user_name, node))
        else:
            pass
    
    '''
    helper function to check if string is in a specific date format
    '''
    def _is_date(self, date_string):
        try:
            date = datetime.datetime.strptime(date_string, '(%m/%d/%Y)')
        except ValueError as err:
            return False
        else:
            return True
        
    '''
    Extracts information from a given line
    
    returns True if data was parsed successfully. False otherwise
    parse_types used to determine what types of data structures to update or skip (currently only linegraph)
    '''
    def _parse_line(self, line_, parse_types = ["linegraph"]):
        # check date, ignore if none (first date was not read yet):
        if self._date is not None:
            if self._end_date is not None and self._date > self._end_date:
                return False
        
        line = line_.strip().split()

        if len(line) < 4: # check if line is empty
            return False
        
        if line[-1] == '(SHUTDOWN)': # special case for shutdown
            line = line[:-1]
        
        # used for case where date is updated after new day's logs are written (starccm.log, line 532)
        
        if (":" in line[0]) and self._prev_line_time > datetime.time(*[int(x) for x in line[0].split(':')]): 
            self._date += datetime.timedelta(days=1) # TODO find better solution. Right now, assuming only 1 day change

        if ':' in line[0]:
            self._prev_line_time = datetime.time(*[int(x) for x in line[0].split(':')]) # used for case where date is updated after new day's logs are written (starccm.log, line 532)

        if line[2] == "OUT:" or line[2] == "IN:": #checkout or checkin
            #print(line)
            numLicense = 1
            pid_default = "[-1]"
            timeStamp, daemon, checkType, feature, user, pid = 0,0,0,0,0,pid_default
            if self._date == None:
                raise Exception("Error, date not known")
            # not all logs have pid, make pid default = -1
            if len(line) == 6: 
                timeStamp, daemon, checkType, feature, user, x = line
                if x == "(SHUTDOWN)":
                    pid = pid_default
                else:
                    pid = x
            elif len(line) == 5: # missing pid value
                timeStamp, daemon, checkType, feature, user = line
            elif len(line) == 8: # has number of licenses checked out
                timeStamp, daemon, checkType, feature, user, x, numLicense, _ = line
                if x == "(SHUTDOWN)":
                    pid = pid_default
                else:
                    pid = x
            elif len(line) == 7: # missing pid line
                timeStamp, daemon, checkType, feature, user, numLicense, _ = line
            else:
                raise Exception("Error, unknown line")

            if self._ignore_pid:
                pid = pid_default
            ## cleaning data
            checkType = checkType[:-1] # remove last character :
            feature = feature[1:-1] # remove first and last characters " "
            pid = pid[1:-1] # remove first and last characters [ ]
            daemon = daemon[1:-1] # remove first and last characters ( )
            if numLicense != 1:
                numLicense = int(numLicense[1:]) # remove first character ( 
            timeStamp = datetime.time(*[int(x) for x in timeStamp.split(':')])#datetime.time.fromisoformat(timeStamp)
            dateTime = datetime.datetime.combine(self._date, timeStamp)

            if self._ignore_numLicense:
                numLicense = 1
            unique_identifier = (feature, user, pid, numLicense)

            #print(f"{dateTime} {checkType} {unique_identifier}")
            self._add_data(checkType, (dateTime, unique_identifier), parse_types)

        elif line[2] == "UNSUPPORTED:": #unsupported license
            pass
            return False
            #print(f"{line[6]} UNSUPPORTED license {line[3]} at {line[0]}")
        elif line[2] == "DENIED:": # either wrong license attempted to be checkout or not enough licenses
            # there are several error messges for denied, we are interested in:
            # (Licensed number of users already reached. (-4,342))
            # (Licensed number of users already reached. (-4,342:104 "Connection reset by peer"))
            # So we check if the string 'reached.' is in index 10 
            # CAREFUL!!! if error message changes this will stop working
            
            if line[10] == "reached.":
                timeStamp = line[0]
                timeStamp = datetime.time(*[int(x) for x in timeStamp.split(':')])#datetime.time.fromisoformat(timeStamp)
                dateTime = datetime.datetime.combine(self._date, timeStamp)
                feature = line[3][1:-1] # remove first and last characters " "
                user = line[4]
                self._add_data('DENIED', (dateTime, feature, user))
            else:
                return False
        elif line[3] == "Start-Date:" or line[3] == "Time:":
            year = int(line[7])
            month = int(monthToInt[line[5]])
            day = int(line[6])
            self._date = datetime.date(year, month, day)
        elif (self._is_date(line[-1])):
            self._date = datetime.datetime.strptime(line[-1], '(%m/%d/%Y)').date()
        elif line[2] == "TIMESTAMP":
            self._date = datetime.datetime.strptime(line[3], '%m/%d/%Y').date()
        elif line[2].lower() == "started" or line[2].lower() == "restarted":
            timeStamp = datetime.time(*[int(x) for x in line[0].split(':')])
            dateTime = datetime.datetime.combine(self._date, timeStamp)
            if (self._start_date is None or self._date >= self._start_date) and (self._end_date is None or self._date <= self._end_date):  
                self._restarts.append(dateTime)
        else:
            return False
        return True

    def parse_file(self, num_lines = None, parse_types = ["linegraph"]):
        num_lines = sum(1 for _ in open(self._log_name))
        log_file = open(self._log_name, 'r', buffering=200000000)
        count = 0
        
        for line in log_file:
            if (count%1024 == 0):
                print(f"{100*count//num_lines}% read (line {count} of {num_lines})", end='\r')
            if self._parse_line(line, parse_types):
                count += 1
            if num_lines != None and count >= num_lines:
                break
    
    def _print_line(self, dateTime, type, feature, user, pid, numLicense):
        if pid != "-1":
            if numLicense==1:
                print(f"{dateTime} {type}: \"{feature}\" {user}  [{pid}]")
            else:
                print(f"{dateTime} {type}: \"{feature}\" {user}  [{pid}]  ({numLicense} licenses)")
        else:
            if numLicense==1:
                print(f"{dateTime} {type}: \"{feature}\" {user}")
            else:
                print(f"{dateTime} {type}: \"{feature}\" {user}  ({numLicense} licenses)")

    '''
    Try to match license checkIn and checkOut pairs
    Note, a few assumptions are made when processing
    '''
    def parse_pairs_old(self, maxPass=3, keyList = []):
        # maxpass = how many times to attempt parsing current key
        
        num_warnings = 0
        num_success = 0
        skipped_intervals = 0
        valid_intervals = 0
        skipped_out = 0
        skipped_in = 0
        total_keys = len(self._licenseChecks)
        iter = -1
        if keyList == None or len(keyList) == 0:
            keyList = self._licenseChecks.keys()
            self._intervals = defaultdict(list) # reset previous intervals if any

        for key in keyList:
            iter += 1
            if iter%1000 == 0:
                print(f"{100*iter//total_keys}% pairs parsed (key {iter} out of {total_keys})", end='\r')
            feature, user, pid, numLicense = key #
            assert('@' in user)
            user_name, node = user.split('@')
                
            checkOutTime = 0
            checkInTime = 0
            skipList = [] # stores skipped over data to attempt to parse later
            ##print(key)
            ##print(self._licenseChecks[key])
            ##print("...")
            for i in range(len(self._licenseChecks[key])):   #"feature_x"  => [(OUT, dateTime), (IN, dateTime), (OUT, dateTime), (IN, dateTime)]
                type, dateTime = self._licenseChecks[key][i]
                if type == "OUT":
                    if checkOutTime == 0:
                        checkOutTime = dateTime
                        if i == len(self._licenseChecks[key])-1: # if last out, append to skip list
                            skipList.append((type, dateTime))
                    else: # ignore subsequent OUT's before IN
                        ##print("WARNING, ignoring subsequent OUTs:")
                        ##self._print_line(dateTime, type, feature, user, pid, numLicense)
                        if maxPass <= 1:
                            num_warnings += 1
                            #skipped_intervals += numLicense
                            skipped_out += 1
                        else:
                            skipList.append((type, dateTime))
                        
                else: # type == "IN"
                    if checkOutTime == 0: # ignore any IN's at the beginning if no OUT came before in log
                        ##print("WARNING, ignoring IN without previous OUT:")
                        ##self._print_line(dateTime, type, feature, user, pid, numLicense)
                        if maxPass <= 1:
                            num_warnings += 1
                            skipped_intervals += numLicense
                            skipped_in += 1
                        else:
                            skipList.append((type, dateTime))
                        
                    else:
                        checkInTime = dateTime
                        delta = datetime.timedelta.total_seconds(checkInTime-checkOutTime)
                        if not (delta >= 0):
                            print(f"checkInTime={checkInTime}  checkOutTime={checkOutTime}")
                            # TODO do this without special case
                            if node == 'comp-ic-0022.acii.production.int.aci.ics.psu.edu' and feature == 'lum_fdtd_solve':
                                print(f"SKIPPING EXCEPTION FOR node:{node} feature:{feature}")
                            else:
                                raise Exception("ERROR, CheckInTime is smaller than CheckOutTime")
                        for i in range(numLicense):
                            self._intervals[feature].append((delta, checkInTime, checkOutTime, user_name, node))
                        valid_intervals += numLicense
                        num_success += 1

                        # reset check times
                        checkOutTime = 0
                        checkInTime = 0
            if maxPass > 1:
                if not ((len(self._licenseChecks[key])-len(skipList))%2 == 0):  # make sure that elements are removed in pairs
                    print("assertion fail")
            # attempt to parse failed lines
            # TODO do this without replicated code
            passes = 1
            earlyFinish = False
            if len(skipList)==0: 
                earlyFinish = True
                ##print(f"EARLY FINISH PASS {passes}")
            while passes<maxPass and not earlyFinish:
                checkOutTime = 0
                checkInTime = 0
                prevLen = len(skipList)
                keyList = skipList.copy()
                skipList = []
                passes += 1
                # temp stats, required when using early finish. TODO make this a dictionary
                _num_warnings = 0
                _skipped_intervals = 0
                _skipped_out = 0
                _skipped_in = 0

                ##print(f"Starting pass {passes}")
                ##print(keyList)
                for i in range(len(keyList)):   #"feature_x"  => [(OUT, dateTime), (IN, dateTime), (OUT, dateTime), (IN, dateTime)]
                    type, dateTime = keyList[i]
                    if type == "OUT":
                        if checkOutTime == 0:
                            checkOutTime = dateTime
                            if i == len(keyList)-1:
                                skipList.append((type, dateTime))
                        else: # ignore subsequent OUT's before IN
                            ##print(f"PASS:{passes} WARNING, ignoring subsequent OUTs:")
                            ##self._print_line(dateTime, type, feature, user, pid, numLicense)
                            if passes == maxPass:
                                num_warnings += 1
                                skipped_out += 1
                            else:
                                _num_warnings += 1
                                _skipped_out += 1
                                skipList.append((type, dateTime))
                            
                    else: # type == "IN"
                        if checkOutTime == 0: # ignore any IN's at the beginning if no OUT came before in log
                            ##print(f"PASS:{passes} WARNING, ignoring IN without previous OUT:")
                            ##self._print_line(dateTime, type, feature, user, pid, numLicense)
                            if passes == maxPass:
                                num_warnings += 1
                                skipped_intervals += numLicense
                                skipped_in += 1
                            else:
                                _num_warnings += 1
                                _skipped_intervals += numLicense
                                _skipped_in += 1
                                skipList.append((type, dateTime))
                            
                        else:
                            checkInTime = dateTime
                            delta = datetime.timedelta.total_seconds(checkInTime-checkOutTime)
                            if not (delta >= 0):
                                print(f"checkInTime={checkInTime}  checkOutTime={checkOutTime}")
                                raise Exception("ERROR, CheckInTime is smaller than CheckOutTime")
                            for i in range(numLicense):
                                self._intervals[feature].append((delta, checkInTime, checkOutTime, user_name, node))
                            valid_intervals += numLicense
                            num_success += 1

                            # reset check times
                            checkOutTime = 0
                            checkInTime = 0
                    # check if any progress was made
                    # if multiple outs at end, first out is not added to skipList TODO: fix
                    #assert(not len(skipList) == prevLen-1)
                if len(skipList)==0 or len(skipList) == prevLen or len(skipList) == prevLen-1: #or len(skipList)==0 , why does adding this give different results?????
                    earlyFinish = True
                    ##print(f"EARLY FINISH PASS {passes}")
                    num_warnings = _num_warnings
                    skipped_intervals = _skipped_intervals
                    skipped_out = _skipped_out
                    skipped_in = _skipped_in
                        
        print(f"!!!error statistics incorrect if maxpass too large. TODO: FIX")
        print(f"number of warnings = {num_warnings}")
        print(f"number of successes = {num_success}")
        print(f"number of skipped license intervals = {skipped_intervals//2}")
        print(f"number of valid license intervals = {valid_intervals}")
        print(f"skipped OUT = {skipped_out}")
        print(f"skipped IN = {skipped_in}")

    
    '''
    Quickly match license intervals too large for bipartite matching.
    '''
    def _parse_remaining_pairs(self, remaining_keys):
        
        # find places to split Check Ins/Outs to reduce problem size
        self.parse_pairs_old(maxPass=100, keyList=remaining_keys)
            
    
    '''
    Match License intervals using min bipartite matching
    '''
    def parse_pairs(self, maxInterval_ = datetime.timedelta(seconds=200000), maxIntervalExceptions: dict = {'hammer': datetime.timedelta(seconds=86400*15)}, includeAllIntervals = True, keyList = [], max_bipartite_matrix_size=10000, split_distance=datetime.timedelta(days=14)):
        # maxpass = how many times to attempt parsing current key
        from scipy.sparse.csgraph import min_weight_full_bipartite_matching, maximum_bipartite_matching
        from scipy.sparse import csr_matrix
        self._intervals = defaultdict(list)
        num_warnings = 0
        num_success = 0
        skipped_intervals = 0
        valid_intervals = 0
        skipped_out = 0
        skipped_in = 0
        successful_splits = 0
        total_added_intervals = 0
        total_keys = len(self._licenseChecks)
        iter = -1
        inf_ = 9999999999.0 # workaround for scipy linear assignment bug when using inf values
        epsilon = 0.1
        skipped_keys = []
        max_len = max_bipartite_matrix_size
        
        if len(keyList) == 0:
            keyList = self._licenseChecks.keys()
        
        for key in keyList:
            added_intervals = 0
            available_intervals = len(self._licenseChecks[key])//2
            if len(self._licenseChecks[key]) > max_len:
                # try reducing problem size
                key_subchecks = []
                _, prev_date_time = self._licenseChecks[key][0]
                prev_index = 0
                for index, data in enumerate(self._licenseChecks[key]):
                    check, date_time = data
                    if date_time-prev_date_time > split_distance:
                        key_subchecks.append(self._licenseChecks[key][prev_index:index])
                        prev_index = index
                    prev_date_time = date_time
                    
                # add last license check interval
                if len(self._licenseChecks[key][prev_index:]) > 0:
                    key_subchecks.append(self._licenseChecks[key][prev_index:])
                
                count = 0
                skip_key = False
                for i in key_subchecks:
                    count += len(i)
                    if len(i) > max_len:
                        print(f"{self._licenseChecks[key]}")
                        print(f"key: {key} cannot be split into sizes smaller than max_len={max_len}, (subinterval of length {len(i)})")
                        skipped_keys.append(key)
                        skip_key = True
                        break
                if skip_key:
                    continue
                if not count == len(self._licenseChecks[key]):
                    assert(count == len(self._licenseChecks[key]))
                successful_splits += 1
            else:
                key_subchecks = [self._licenseChecks[key]]
            iter += 1
            if iter%1 == 0:
                print(f"{100*iter//total_keys}% pairs parsed (key {iter} out of {total_keys}), skipped = {len(skipped_keys)}, splits = {successful_splits}", end='\n')
            feature, user, pid, numLicense = key
            assert('@' in user)
            user_name, node = user.split('@')
            #print(f"Number of checks = {len(self._licenseChecks[key])}, subchecks = {len(key_subchecks)}")
            #print(f"node {node}")
            # older nodes (e.g. hammer) can have a longer job duration, make exceptions for these
            maxInterval = maxInterval_
            for nodeName in maxIntervalExceptions.keys():
                if nodeName in node:
                    maxInterval = maxIntervalExceptions[nodeName]
                    #print(f"{nodeName} in node")
                    break
                    
            for checks in key_subchecks:
                checkOutTime = 0
                checkInTime = 0
                outList = [] # temporarily stores skipped outs
                inList = []
                
                for i in range(len(checks)):   #"feature_x"  => [(OUT, dateTime), (IN, dateTime), (OUT, dateTime), (IN, dateTime)]
                    type, dateTime = checks[i]
                    if type == "OUT":
                        outList.append(dateTime)  
                    else: # type == "IN"
                        inList.append(dateTime)
                        
                if len(outList) != len(inList):
                    print(f"num CheckOut [{len(outList)}] and num CheckIn [{len(inList)}] not equal for {feature} {user}")
                            

                assert(outList == sorted(outList))
                assert(inList == sorted(inList))
                # init graph biadjacency matrix representation
                #matrix = np.zeros((len(outList), len(inList)), dtype=np.float64)
                ## print(f"({len(outList)}, {len(inList)})")
                #csr_list = []
                csr_weight_list = []
                csr_row_list = []
                csr_col_list = []
                time1 = datetime.datetime.now()
                for outInd, checkOut in enumerate(outList): # loop is slow TODO improve
                    for inInd, checkIn in enumerate(inList):
                        if checkIn < checkOut:
                            continue
                        if checkIn-checkOut > maxInterval:
                            break
                        else:
                            checkIn: datetime.datetime
                            checkOut: datetime.datetime
                            weight = (checkIn-checkOut).total_seconds()
                            #print(f"weight={weight}  checkin={checkIn}, checkout={checkOut}")
                            assert (weight>=0)
                            if weight == 0: # 0 weights cause issues in min_weight_full_bipartite_matching
                                weight = epsilon
                            #matrix[outInd, inInd] = weight
                            #csr_list.append((weight, outInd, inInd))
                            csr_weight_list.append(weight)
                            csr_row_list.append(outInd)
                            csr_col_list.append(inInd)
                ## print(f"csr_list time {datetime.datetime.now()-time1}")
                #if len(outList) < 50:
                #    print(outList)
                #    print(inList)
                print(f"{feature} {user}")
                # add column of inf to ensure full matching
                time2 = datetime.datetime.now()
                if len(csr_weight_list) > 0:
                    weights, rows, cols = csr_weight_list, csr_row_list, csr_col_list
                    #print('performing bipartite matching')
                    bipartite_matching = maximum_bipartite_matching( csr_matrix((weights, (rows, cols)), shape=((len(outList), len(inList)))) )
                    #print('performing bipartite matching - Done')
                    cardinality = (bipartite_matching >= 0).sum() # count how many matches
                    #_ = csr_matrix((weights, (rows, cols)), shape=((len(outList), len(inList)))).toarray()
                        
                    num_zero_rows = len(set([i for i in range(len(outList))]) - set(rows))
                    #num_zero_cols = np.sum(~_.T.any(1))
                    num_zero_cols = len(set([i for i in range(len(inList))]) - set(cols))
                    num_inf_columns_pred = max(len(outList)-len(inList), 0) + num_zero_rows+num_zero_cols + 1
                else:
                    cardinality = len(outList)
                    num_inf_columns_pred = max(len(outList)-len(inList), 0)
                
                ##print(f"inf column estimate time = {datetime.datetime.now()-time2}")
                ##print(f"num_inf_columns_pred = {num_inf_columns_pred}")
                ##print(f"ground truth = {max(len(outList)-cardinality,0)}")
                #print(f"Inf rows ESTIMATE = {num_inf_columns} vs {max(len(outList)-cardinality, 0)}")
                
                num_inf_columns = max(len(outList)-cardinality, num_inf_columns_pred)
                time3 = datetime.datetime.now()
                for i in range(len(outList)):
                    for j in range(num_inf_columns):
                        #csr_list.append((inf_, i, len(inList)+j))
                        csr_weight_list.append(inf_)
                        csr_row_list.append(i)
                        csr_col_list.append(len(inList)+j)
                ##print(f"adding inf colum time {datetime.datetime.now()-time3}")
                
                # perform minimum weighted bipartite matching
                if len(csr_weight_list) > 0:
                    time3_1 = datetime.datetime.now()
                    weights, rows, cols = csr_weight_list, csr_row_list, csr_col_list
                    ##print(f"unzip time {datetime.datetime.now() - time3_1}")
                    time3_2 = datetime.datetime.now()
                    matrix = csr_matrix((weights, (rows, cols)), shape=((len(outList), len(inList)+num_inf_columns)) )
                    ##print(f"convert to csr matrix time {datetime.datetime.now()-time3_2}")
                    _ = matrix.toarray().tolist()
                    #if len(outList) < 60:
                    #    for a in _:
                    #        print(a)
                    time4 = datetime.datetime.now()
                    outMatchInd, inMatchInd = min_weight_full_bipartite_matching(matrix)
                    ##print(f"min bipartite matching time {datetime.datetime.now()-time4}")
                    time5 = datetime.datetime.now()
                    matrix = matrix.toarray()
                    ##print(f"matrix to array (second time) {datetime.datetime.now()-time5}")
                    
                # add valid matches to self._intervals
                time6 = datetime.datetime.now()
                infOutInd = []
                infInInd = []
                for ind in zip(outMatchInd, inMatchInd):
                    if len(csr_weight_list) == 0 or ind[1] >= len(inList): # check if infinity column(s) matched
                        infOutInd.append(ind[0])
                        infInInd.append(ind[1])
                        continue
                    if matrix[ind[0]][ind[1]] >= inf_:
                        assert(False)
                        infOutInd.append(ind[0])
                        infInInd.append(ind[1])
                        continue
                    
                    checkOutTime, checkInTime = outList[ind[0]], inList[ind[1]]
                    delta = datetime.timedelta.total_seconds(checkInTime-checkOutTime)
                    
                    assert(delta <= maxInterval.total_seconds() )
                    assert(delta == matrix[ind[0]][ind[1]] or (delta == 0.0 and matrix[ind[0]][ind[1]] == epsilon))
                    if not (delta >= 0):
                        print(f"checkInTime={checkInTime}  checkOutTime={checkOutTime}")
                        raise Exception("ERROR, CheckInTime is smaller than CheckOutTime")
                    for i in range(numLicense):
                        self._intervals[feature].append((delta, checkInTime, checkOutTime, user_name, node))
                        added_intervals += 1
                        
                # deal with unmatched check-ins or check-outs
                unmatchedOut = []
                unmatchedIn = []
                for i in range(len(outList)):
                    if i not in outMatchInd or i in infOutInd: # TODO is i not in outMatchInd necessary to check?
                        unmatchedOut.append(outList[i])
                #print(f"unmatched outs: {unmatchedOut}")
                
                for i in range(len(inList)):
                    if i not in inMatchInd or i in infInInd:
                        unmatchedIn.append(inList[i])
                #print(f"unmatched ins: {unmatchedIn}")
                
                # Find next log restart for each CheckOut and assume that it contains a CheckIn
                for checkOut in unmatchedOut:
                    next_restart = None
                    for restartDate in self._restarts: # TODO make this a binary search
                        if restartDate > checkOut:
                            next_restart = restartDate
                            print(f"Next restart = {next_restart}")
                            break
                        
                    if next_restart == None:
                        print (f"WARNING: No next restart for CheckOutTime {checkOut} for {feature} {user}")
                        continue
                    
                    delta = datetime.timedelta.total_seconds(next_restart-checkOut)
                    if not (delta >= 0):
                        print(f"next_restart={next_restart}  checkOutTime={checkOut}")
                        raise Exception("ERROR, CheckInTime is smaller than CheckOutTime")
                    
                    if (delta > maxInterval.total_seconds() and delta < inf_):
                        print(f"WARNING: Delta {delta} too large for unmatched CheckOut at {checkOut} for {feature} {user}")
                        if includeAllIntervals: # TODO need to check if using restart better or worse than directly matching
                            for i in range(numLicense):
                                self._intervals[feature].append((delta, next_restart, checkOut, user_name, node))
                                added_intervals += 1
                    else:
                        for i in range(numLicense):
                            self._intervals[feature].append((delta, next_restart, checkOut, user_name, node))
                            added_intervals += 1
            
                
                # Find previous log restart for each CheckIn and assume that it contains a CheckOut
                for checkIn in unmatchedIn:
                    prev_restart = None
                    for restartDate in reversed(self._restarts):
                        if restartDate < checkIn:
                            prev_restart = restartDate
                            break
                    if next_restart == None:
                        #raise Exception(f"No Prev restart for CheckInTime {checkIn} for {feature} {user}")
                        print(f"WARNING: No Prev restart for CheckInTime {checkIn} for {feature} {user}")
                        continue
                    
                    delta = datetime.timedelta.total_seconds(checkIn-prev_restart)
                    if not (delta >= 0):
                        print(f"checkIn={next_restart}  prev_restart={checkOut}")
                        raise Exception("ERROR, CheckInTime is smaller than CheckOutTime")
                    
                    if (delta > maxInterval.total_seconds()):
                        print(f"WARNING: Delta {delta} too large for unmatched CheckIn at {checkIn} for {feature} {user}")
                        if includeAllIntervals:
                            for i in range(numLicense):
                                self._intervals[feature].append((delta, next_restart, checkOut, user_name, node))
                                added_intervals += 1
                    else:
                        for i in range(numLicense):
                            self._intervals[feature].append((delta, checkIn, prev_restart, user_name, node))
                            added_intervals += 1
                ##print(f"rest of matching time {datetime.datetime.now()-time6}")
            total_added_intervals += added_intervals
            if added_intervals < available_intervals*0.8:
                print(f"WARNING: Intervals added for key {key} : ({added_intervals} vs {available_intervals})")
                
        ## end interval for
        
        print(skipped_keys)
        print(f"number of skipped keys = {len(skipped_keys)}")
        print(f"number of added intervals = {added_intervals}")
        skipped_intervals = 0
        for k in skipped_keys:
            skipped_intervals += len(self._licenseChecks[k])
        print(f"skipped intervals = {skipped_intervals}")
        #print("parsing skipped keys")
        #self._parse_remaining_pairs(skipped_keys)
        return skipped_keys

    '''
    Used for plotting interarrival times for each feature
    '''    
    def get_interarrival_times(self, plot="bar", font_size=None):
        features = [feature for feature in self._intervals]
        if plot == "bar":
            means = []
            stds = [] # standard deviations
            x_pos = np.arange(len(features))
            for feature in features:
                feature_times = np.array([time[0] for time in self._intervals[feature]])
                means.append(np.mean(feature_times))
                stds.append(np.std(feature_times))
            
            # sort lists in descending order
            zipped = list(zip(means, stds, features))
            zipped.sort(reverse=True, key=lambda x: x[0])
            means, stds, features = zip(*zipped)

            # Build the plot
            fig, ax = plt.subplots()
            ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)
            ax.set_ylabel('mean license use time (s)')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(features)
            ax.set_title(f'{self._name} License Feature Usage Times')
            ax.yaxis.grid(True)
            plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')

            # Save the figure and show
            print(f"Displaying {len(features)} features")
            plt.tight_layout()
            plt.savefig(self._name+'_license_means_bar_plot_with_error_bars.png')
            plt.show()
        elif plot == "box" or plot == "box_nostrip":
            data = []
            means = []
            x_pos = np.arange(len(features))
            for feature in features:
                feature_times = np.array([time[0] for time in self._intervals[feature]])
                data.append(feature_times)
                means.append(np.mean(feature_times))

            # add number of data points information
            feature_lens = []
            for i in range(len(features)):
                features[i] += str(f" [{len(data[i])}]")
                feature_lens.append(len(data[i]))

            # sort lists in descending order
            '''
            zipped = list(zip(means, data, features))
            zipped.sort(reverse=True)
            means, data, features = zip(*zipped)
            '''
            zipped = list(zip(feature_lens, means, data, features))
            zipped.sort(reverse=True, key=lambda x: x[0]) # define key, otherwise may get error is keys are same and other elements are compared
            feature_lens, means, data, features = zip(*zipped)

            plot_data = list(zip(data))

            
            ax = sns.boxplot(data=data, showfliers = False)
            if plot != "box_nostrip":
                ax = sns.stripplot(data=data, edgecolor="black", marker="o", linewidth=1, jitter=0.3, alpha=0.5, s=5)
            ax.set_xticklabels(features)
            ax.set_ylabel('mean license use time (s)')
            if self._name != None and self._name != '':
                ax.set_title(f'{self._name} License Feature Use Interval Times')
            plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
            if font_size != None:
                plt.rcParams['font.size'] = font_size
            plt.savefig(self._name+'_license_box_plot_.pdf', format="pdf", bbox_inches='tight')
            plt.show()
            
        elif plot == "violin":
            data = []
            means = []
            x_pos = np.arange(len(features))
            for feature in features:
                feature_times = np.array([time[0] for time in self._intervals[feature]])
                data.append(feature_times)
                means.append(np.mean(feature_times))

            # add number of data points information
            for i in range(len(features)):
                features[i] += str(f" [{len(data[i])}]")

            # sort lists in descending order
            zipped = list(zip(means, data, features))
            zipped.sort(reverse=True, key=lambda x: x[0])
            means, data, features = zip(*zipped)

            plot_data = list(zip(data))

            
            ax = sns.violinplot(data=data, scale = "width")
            ax = sns.stripplot(data=data, edgecolor="black", marker="o", linewidth=1, jitter=0.3, alpha=0.5, s=5)
            ax.set_xticklabels(features)
            ax.set_ylabel('mean license use time (s)')
            ax.set_title(f'{self._name} License Feature Use Interval Times')
            plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
            plt.savefig(self._name+'_license_violin_plot_.png')
            plt.show()

            '''
            fig, ax = plt.subplots()
            #ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)
            ax.set_ylabel('license use time (s)')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(features)
            ax.set_title(f'{self._name} License Feature Usage Times')
            ax.yaxis.grid(True)
            plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='right')

            print(f"Displaying {len(features)} features")
            plt.tight_layout()
            plt.boxplot(data, labels=features)
            plt.savefig(self._name+'_license_box_plot.png')
            plt.show()
            '''
    '''
    Used for plotting line graph of how much features used at a given time
    '''
    def get_line_graph(self, type = 'seaborn'):
        data = pd.concat([self._lineGraphData], axis=0, ignore_index=True)

        print(data)
        print(data)
        print("updating index. Why does this take so long??")
        #self._lineGraphData = self._lineGraphData.set_index("dateTime") # why does this take so long?
        #self._lineGraphData = self._lineGraphData.reset_index(drop=True)

        if type == "seaborn":
            ax = sns.lineplot(data)
            ax.set_ylabel('Number of checked out licenses')
            ax.set_xlabel('time')
            #ax.set_xticklabels([datetime.fromtimestamp(tm) for tm in ax.get_xticks()], rotation=50)
            ax.set_title(f"{self._name} License Check Outs")
            plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
            plt.savefig('./plots/'+self._name+'_license_line_plot.png')
            plt.show()
        else:
            print(data)
            data = data.set_index("dateTime")
            fig = px.line(data, title=f"{self._name} License Check Outs",  template="seaborn",
                    labels = {
                        "value" : "Number of Check-outs"
                    })
            #ax.set_ylabel('Number of checked out licenses')
            #ax.set_xlabel('time')
            #plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
            fig.show()
    
    '''
    Save important data (license checkIn + checkOut pairs & denied licenses) to a file
    '''
    def save_to_file(self, filename, data_type):
        # save license pairs
        if data_type == 'pair':
            f = open(filename, "w")
            f.write("feature_names,delta,check_out,check_in,user_name,node\n")
            first_line = True
            for feature_name in self._intervals:
                for delta, checkIn, checkOut, user_name, node in self._intervals[feature_name]:
                    if first_line:
                        first_line = False
                    else:
                        f.write("\n")
                    node = node.split('.')[0] # get rid of suffix, since in accounting logs the suffix is omitted
                    f.write(f"{feature_name},{delta},{checkOut},{checkIn},{user_name},{node}")
            f.close()
        
        # save license denieds
        if data_type == 'denied':
            f = open(filename, "w")
            f.write("feature_names,denied_dateTime,user_name,node\n")
            for feature_name in self._denied:
                for denied_dateTime, user_name, node in self._denied[feature_name]:
                    node = node.split('.')[0] # get rid of suffix, since in accounting logs the suffix is omitted
                    f.write(f"{feature_name},{denied_dateTime},{user_name},{node}\n")
            f.close()




#%%
if __name__ == "__main__":  
    
    #print(f"current directory: {os.getcwd()}")
    log_name = "D:\logs\lm\comsol\comsol60.log_deidentified"#"D:\logs\lm\matlab\matlab.log_deidentified"#"./logs/starccm.log" #    
    name = "comsol_new_jan" # should not contain spaces
    my_license = License(log_name, name, ignore_pid=False, ignore_numLicense=False)
    #%%
    # using linegraph is VERY slow TODO fix this
    my_license.parse_file(parse_types = []) #parse_types = ["linegraph"]
    #%%
    skipped_keys = my_license.parse_pairs_old(maxPass=100)
    #skipped_keys = my_license.parse_pairs(split_distance=datetime.timedelta(days=7), includeAllIntervals=False)
    #%%
    my_license._parse_remaining_pairs(skipped_keys)
    #%%
    my_license.save_to_file(f"./data/{name}.csv", 'pair')
    my_license.save_to_file(f"./data/{name}_denied.csv", 'denied')
    #%%
    # WARNING: drawing plots is very slow
    my_license.get_interarrival_times(plot="box_nostrip", font_size=28)
    #my_license.get_interarrival_times(plot="box", font_size=28)
    #my_license.get_line_graph(type="seaborn")
    #%%
    
    font_size = 22#28
    width = 10*2
    height = 6*2
    data = []
    means = []
    features = [feature for feature in my_license._intervals]
    x_pos = np.arange(len(features))
    
    remove_outliers = True
    max_time = 175000
    
    for feature in features:
        feature_times = np.array([time[0] for time in my_license._intervals[feature]])
        if remove_outliers:
            feature_times[feature_times > max_time] = max_time
        data.append(feature_times)
        means.append(np.mean(feature_times))

    # add number of data points information
    feature_lens = []
    for i in range(len(features)):
        if len(features[i])>11:
            features[i] = features[i][:9]+'...'
        features[i] += str(f" [{len(data[i])}]")
        feature_lens.append(len(data[i]))

    # sort lists in descending order
    '''
    zipped = list(zip(means, data, features))
    zipped.sort(reverse=True)
    means, data, features = zip(*zipped)
    '''
    zipped = list(zip(feature_lens, means, data, features))
    zipped.sort(reverse=True)
    feature_lens, means, data, features = zip(*zipped)

    plot_data = list(zip(data))

    plt.rcParams["figure.figsize"]=(width, height)
    if font_size != None:
        plt.rcParams['font.size'] = font_size

    ax = sns.boxplot(data=data, showfliers = False)
    ax = sns.stripplot(data=data, edgecolor="black", marker="o", linewidth=1, jitter=0.3, alpha=0.5, s=5)

    ax.set_xticklabels(features)
    ax.set_ylabel('license use time (s)')
    if my_license._name != None and my_license._name != '':
        ax.set_title(f'{my_license._name} License Feature Use Interval Times')
    plt.setp(ax.get_xticklabels(), rotation=60, horizontalalignment='right')
    if font_size != None:
        plt.rcParams['font.size'] = font_size
    plt.rcParams["figure.figsize"]=(width, height)
    plt.savefig(name+'_license_box_plot_.pdf', format="pdf", bbox_inches='tight')
    
# %%
