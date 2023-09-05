import unittest
import simulator.simulator as s
import datetime
from typing import Callable

class BasicForecast(s.ForecastModel):
    
    def __init__(self, base_delay: s.DurationType) -> None:
        super().__init__()
        self._history = {}
        self._delay = base_delay
    
    def init_stats(self, stats: s.Statistics) -> None:
        pass
    
    def forecast(self, license_name: s.LicenseName, cur_time: s.TimeType, max_delay: s.TimeType, min_license: int, window_size: s.Tuple[s.TimeType, s.TimeType], job_id: str=None) -> s.TimeType:
        '''
        license_name: Name of license to forecast
        cur_time: start time of forecasting
        max_delay: maximum delay time for license
        min_license: return forecast time where minimum license is less than or equal to this amount
        window_size: size of min_license window, e.g. if window_size=(2min, 2min), return data point where at least previous 2 minutes and at least subsequent 2 minutes are less than or equal to min_license. NOTE: granularity depends on model used, if model has 1 minute granularity, using a window size of (10sec, 10sec) will round UP to 1 minute granularity
        
        returns: how long to delay license/job for
        '''
        if self._history.get(job_id) == None:
            self._history[job_id] = 0
        self._history[job_id] +=1
        return (2**(self._history[job_id]-1))*self._delay
    
class DistributionForecast(s.ForecastModel):
    import operator
    import functools
    import pandas as pd
    import numpy as np
    from math import ceil, isnan
    def __init__(self, base_delay: s.DurationType, min_license_length: datetime.timedelta, max_license_length: datetime.timedelta, license_interval_file: str, toDurationType: Callable[[int], s.DurationType] = None, windowLength: s.DurationType = None, windowSize: int = None, resubmissionDelay: s.DurationType = 0) -> None:
        super().__init__()
        self._history = {}
        #self._delay: int = int(base_delay.total_seconds())
        self._stats: s.Statistics = None
        self._max_license_length: int = int(max_license_length.total_seconds())
        self._min_license_length: int = int(min_license_length.total_seconds())#max(int(min_license_length.total_seconds()), int(resubmissionDelay.total_seconds()))
        self._to_duration_type = toDurationType
        self._time_granularity = 1 # in seconds #TODO add this as parameter
        self._increasing = self.pd.Series([i for i in range(int(self._max_license_length/self._time_granularity))])
        ###
        self._dist_data = self.pd.read_csv(license_interval_file, parse_dates=[2,3]) #TODO update this to use only past data
        self._window_length = windowLength
        self._window_size = windowSize

    def forecast(self, license_name: s.LicenseName, cur_time: s.TimeType, max_delay: s.TimeType, min_license: int, window_size: s.Tuple[s.TimeType, s.TimeType], job_id: str=None, precision = np.longdouble, DEBUG_TEST=False, DEBUG_PRINT=False, DEBUG_PLOT=False) -> s.DurationType:
        '''
        Wrapper for _forecast function
        
        license_name: Name of license to forecast
        cur_time: start time of forecasting
        max_delay: maximum delay time for license
        min_license: return forecast time where minimum license is less than or equal to this amount
        window_size: size of min_license window, e.g. if window_size=(2min, 2min), return data point where at least previous 2 minutes and at least subsequent 2 minutes are less than or equal to min_license. NOTE: granularity depends on model used, if model has 1 minute granularity, using a window size of (10sec, 10sec) will round UP to 1 minute granularity
        
        returns: how long to delay license/job for
        '''

        if self._window_length != None:
            result = self._forecast(license_name, cur_time, max_delay, min_license, window_size, job_id, precision, DEBUG_TEST, DEBUG_PRINT, DEBUG_PLOT, window_length=self._window_length)
        else:
            result = self._forecast(license_name, cur_time, max_delay, min_license, window_size, job_id, precision, DEBUG_TEST, DEBUG_PRINT, DEBUG_PLOT)
            
        return result
        
    
    # TODO add case when license already exists as part of job - ignore those licenses?? CHECK if licenses checked out BEFORE delay added        
    def _forecast(self, license_name: s.LicenseName, cur_time: s.TimeType, max_delay: s.TimeType, min_license: int, window_size: s.Tuple[s.TimeType, s.TimeType], job_id: str=None, precision = np.longdouble, DEBUG_TEST=False, DEBUG_PRINT=False, DEBUG_PLOT=False, window_length = None) -> s.DurationType:
        '''
        license_name: Name of license to forecast
        cur_time: start time of forecasting
        max_delay: maximum delay time for license
        min_license: return forecast time where minimum license is less than or equal to this amount
        window_size: size of min_license window, e.g. if window_size=(2min, 2min), return data point where at least previous 2 minutes and at least subsequent 2 minutes are less than or equal to min_license. NOTE: granularity depends on model used, if model has 1 minute granularity, using a window size of (10sec, 10sec) will round UP to 1 minute granularity
        
        returns: how long to delay license/job for
        '''

        dist = self._get_dist(license_name, min_val=self._min_license_length, max_val=self._max_license_length, cur_time=cur_time, window_length=window_length) # TODO change this to get data from stats (add new parameter to stats init for historical license usage)
        
        if DEBUG_PRINT: print(f"dist {dist}")
        if DEBUG_TEST:
            T_ = [int(i) for i in input('current license use times: ').split(',')]
        else:
            T_ = [int(i.total_seconds()) for i in self._stats.getLicenseUseTimes(license_name, cur_time)]
        
        print(f"T_ = {T_} \nlength = {len(T_)}\nlicense name = {license_name}")
        n = len(T_)
        assert(n>0)
        #prob_x_greater = [self._prob_greater(dist, y) for y in range(self._max_license_length)]
        #prob_x_greater = (dist.iloc[::-1].cumsum()[::-1]).astype(precision)/dist.sum()
        prob_x_greater = (dist.iloc[::-1].cumsum()[::-1]).astype(precision) # no need to divide by dist.sum() 
                                                                            # since both numerator and denominator 
                                                                            # has this value in the expected delay formula
                                                                            # This also increases numerical stability significantly
        if DEBUG_PRINT: print(f"prob_x_greater: \n{prob_x_greater}")
        
        
        max_remaining_runtime = self._max_license_length - max(T_)
        assert(type(max_remaining_runtime) == int)
        assert(max_remaining_runtime > 0)
        
        
        if DEBUG_PRINT:
            time2 = datetime.datetime.now()
            F_y_old = self.pd.Series([1.0]*(self._max_license_length+1))
            for y in range(max_remaining_runtime+1): # calculate cdf
                F_y_old[y] = 1.0 - self._mult_reduce([prob_x_greater[y+T_[i]] for i in range(n)])
            print(f"====for loop method time {datetime.datetime.now()-time2}")
            
        if DEBUG_PRINT: time1 = datetime.datetime.now()
        
        # initialize F_y values, store index value for indexes that will be changed
        F_y = self.pd.Series( [i for i in range(max_remaining_runtime+1)] + [1.0] *(self._max_license_length-max_remaining_runtime))
        cdf_func = lambda y: 1.0 - self._mult_reduce_vectorized(self.np.column_stack([prob_x_greater[y+T_[i]] for i in range(n)]))
        print(f"max_remaining_runtime = {max_remaining_runtime}")
        F_y.iloc[0:max_remaining_runtime+1] = cdf_func(F_y.iloc[0:max_remaining_runtime+1].values)
        ######################################################################
        if DEBUG_PRINT: 
            print(f"====vector method time {datetime.datetime.now()-time1}")

            #print(f"lengths {len(F_y.iloc[0:max_remaining_runtime+1])} vs {len(f(F_y.iloc[0:max_remaining_runtime+1].values))} vs {len(F_y.iloc[0:max_remaining_runtime+1].values)}")
            print(f"F_y_old \n{F_y_old}")
            print(f"F_y \n{F_y}")
            
            print('COMPARE')
            comp = F_y_old.compare(F_y)
            #print(comp)
            assert(len(comp)==0)
            print('===')
        
        f_y = F_y.diff() # get pmf by differentiating cdf
        f_y[0] = 0.0 # replace NaN value
        if DEBUG_PRINT: 
            print(f"f_y \n{f_y}")
            if DEBUG_PLOT:
                import matplotlib.pyplot as plt
                f_y.plot()
                plt.show()
                F_y.plot()
                plt.show()
            
        
        
        integral = self._integrate(f_y*self._increasing, start=0, end=max_remaining_runtime)

        E_time_until_next_checkin = integral/self._mult_reduce([prob_x_greater[T_[i]] for i in range(n)])
        print(f"numerator = {integral}, denominator = {self._mult_reduce([prob_x_greater[T_[i]] for i in range(n)])}")
        if DEBUG_PRINT: print(f"E_time={E_time_until_next_checkin}, integral={integral}, condition = {1/self._mult_reduce([prob_x_greater[T_[i]] for i in range(n)])}")

        if (self.isnan(E_time_until_next_checkin)):
            E_time_until_next_checkin = 0.0
            
        '''
        for y in range(0, max_remaining_runtime):
            cdf_y = 1
            cdf_y_plus_1 = 1
            for i in range(n): # take derivative
                cdf_y_plus_1 *= self._prob_greater(dist, y+1+T[i])/prob_x_greater_t1[i]
                cdf_y *= self._prob_greater(dist, y+T[i])/prob_x_greater_t1[i]
            cdf_dy = cdf_y_plus_1 - cdf_y
            integral += y*cdf_dy
        delay = integral/self.functools.reduce(self.operator.mul, T, 1)
        '''
        return self._to_duration_type(self.ceil(E_time_until_next_checkin))
    
    def _get_dist(self, license_name: str, feature_name_column = 'feature_names', interval_length_column = 'delta', min_val=1, max_val=180000, window_length = None, check_in_name_column = 'check_in', cur_time: s.TimeType = None) -> pd.Series:
        if license_name == 'UNIFORM':
            return self.pd.Series([1.0]*(max_val+1))
                    
        data = self._dist_data.query(f"`{feature_name_column}` == @license_name and @min_val <= `{interval_length_column}` <= @max_val")
        
        # make sure future data is not used
        if cur_time != None:
            data = data.query(f"`{check_in_name_column}` <= @cur_time")
            
        # use data only within a time window
        if window_length != None:
            data_start_time = cur_time - window_length
            data = data.query(f"`{check_in_name_column}` > @data_start_time")
            
        print(f"distribution num data points = {len(data)}")    
        assert(len(data) > 0)
        
        dist = data[interval_length_column].sort_values(ascending=True, ignore_index=True)
        dist = dist.value_counts().reindex(list(range(0,max_val+1)), fill_value=0)
        return dist
    
    def _duration_type_ceil(self, time: s.DurationType) -> s.DurationType:
        return self.pd.Timedelta.ceil(time, freq='s')
        
    def _mult_reduce(self, data: list) -> float:
        if len(data)==1:
            return data[0]
        return self.functools.reduce(self.operator.mul, data, 1)
    
    def _mult_reduce_vectorized(self, data: np.array) -> np.array:
        return self.np.multiply.reduce(data, axis=1)#self.np.prod(data, axis=1)
    
    def _derive(self, data: pd.Series) -> pd.Series:
        '''
        Performs a numerical derivation on parameter data
        
        data: data to perform derivation on
        
        returns: derived data
        '''
        return data.diff(periods=1)
    
    def _integrate(self, data: pd.Series, start: int, end: int) -> pd.Series:
        '''
        Performs a numerical integration on parameter data
        
        data: data to perform integration on
        
        returns: integrated data
        '''
        
        return data.iloc[start:end+1].sum()
    
    def _prob_greater(self, dist: pd.Series, value: int) -> float:
        '''
        Calculates Pr(X > value) where X ~ dist
        
        dist: probability density of X (license runtimes)
        value: threshold to compare X against
        '''
        
        return dist.gt(value).sum()/dist.sum()
    
    
def read_max_license_file(fname, multiple_license_versions_behavior = 'add'):
    
    with open(fname, 'r') as f:
        license_max = {}
        f.readline()
        f.readline()
        f.readline()
        f.readline()
        for l in f:
            row = [i.strip() for i in l.split(' ') if i!='']
            print(row)
            f.readline()

            name = row[0]
            version = float(row[1])
            max_amount = int(row[2])
            if max_amount == 0:
                max_amount = 9999
            if license_max.get(name) == None:
                license_max[name] = max_amount
            else:
                if multiple_license_versions_behavior == 'add':
                    license_max[name] += max_amount
                if multiple_license_versions_behavior == 'max':
                    license_max[name] = max(max_amount, license_max[name])
                if multiple_license_versions_behavior == 'min':
                    license_max[name] = min(max_amount, license_max[name])
        return license_max



if __name__ == "__main__":
    #unittest.main()
    
    
    
    def toTimeType(string: str) -> s.TimeType:
        return datetime.datetime.strptime(string, '%Y-%m-%d %H:%M:%S')
    def toDurationType(string: str) -> s.DurationType:
        return datetime.timedelta(seconds=int(string))
    
    license_names = ['LICENSE1', 'LICENSE2', 'LICENSE3']
    license_amounts = [3,3,3]
    #license_names = ['SERIAL', 'COMSOL', 'COMSOLGUI', 'CORROSION', 'COMSOLUSER', 'COMSOLCKL', 'COMSOLGUICKL', 'STRUCTURALMECHANICSCKL', 'GEOMECHANICSCKL', 'STRUCTURALMECHANICS', 'HEATTRANSFER', 'CFD', 'ACOUSTICS', 'PARTICLETRACING', 'MEMS', 'NONLINEARSTRUCTMATERIALS', 'CADIMPORT', 'CADIMPORTUSER', 'LLMATLABCKL', 'CLUSTERNODE', 'CHEM', 'MICROFLUIDICS', 'PLASMA', 'RF', 'RFCKL', 'ACDC', 'LLMATLAB', 'WAVEOPTICS', 'BATTERIESANDFUELCELLS', 'BATTERYDESIGN', 'SUBSURFACEFLOW', 'CFDCKL', 'SEMICONDUCTOR', 'MULTIBODYDYNAMICS', 'CADREADER', 'ECADIMPORT']
    no_denial_licenses = ['CLUSTERNODE'] # licenses that should/do not have denials in historical data, but do in sumulation due to mistakes in license to job matching, or incorrect max license
    ignored_licenses = ['CLUSTERNODE', 'RFCKL', 'SERIAL']
    licenses = {}
    for l, amount in zip(license_names, license_amounts):
        licenses[l] = amount
    
    licenses = read_max_license_file('./max_licenses.txt', multiple_license_versions_behavior='max')
    for l in no_denial_licenses:
        licenses[l] = 9999
    
    max_compute_nodes = 999999#6630#30000
    resubmission_delay: s.DurationType = toDurationType(60*10)
    max_job_length: s.DurationType = toDurationType(200000)#toDurationType(310000)#toDurationType(180000)
    min_license_length: s.DurationType = toDurationType(1)
    
    job_file_name = './processed/simulator/comsol2015_jobs_2017-10-01-2017-11-03.csv'#'./processed/simulator/comsol2015_jobs.csv'
   
    #job_file_name = './simulator/test_jobs.txt'
    use_historic_queue_time = True
    license_interval_file = './data/comsol2015.csv'
    results_file = './tmp/sim_results'
    #base_delay = toDurationType('300')
    queue_names = ['batch', 'open']
    
    
    # initialize classes
    job_stream = s.JobStream(job_file_name, queue_names, sort=True, toTimeType=toTimeType, toDurationType=toDurationType, use_historic_queue_time=use_historic_queue_time, ignored_licenses=ignored_licenses, max_job_duration=max_job_length)
    
    
    ### FORECAST MODELS ###
    #forecast_model = None
    #forecast_model = BasicForecast(base_delay=resubmission_delay)
    forecast_model = DistributionForecast(None, min_license_length, max_job_length, license_interval_file, toDurationType=toDurationType, resubmissionDelay = resubmission_delay)

    # Distribution forecast with window
    days = 14
    window_length = toDurationType(days*24*60*60)
    #forecast_model = DistributionForecast(None, min_license_length, max_job_length, license_interval_file, toDurationType=toDurationType, resubmissionDelay = resubmission_delay, windowLength=window_length)

    test_forecast = False
    if test_forecast:
        DEBUG_PRINT = False
        forecast_model.init_stats(s.Statistics())
        l_name = input('Feature name: ')
        while l_name != 'q':
            delay = forecast_model.forecast(l_name, cur_time=None, max_delay=0, min_license=None, window_size=None, job_id=None, DEBUG_TEST=True, DEBUG_PRINT=DEBUG_PRINT)
            print(f"delay = {delay}  [{delay.total_seconds()} s]")
    
    sim = s.Simulator(
        queue_names=queue_names,
        licenses=licenses,
        max_compute_nodes=max_compute_nodes,
        job_stream=job_stream,
        resubmission_delay=resubmission_delay,
        forecast_model=forecast_model,
        forecast_parameters=(1,1,(1,1)),
        use_historic_queue_time=use_historic_queue_time
    )
    
    
    
        
    
    sim.run()
    
    stats = sim.getStats()
    
    stat_data = sim.getStatData()
    
    _,_,_,_, forecast_accuracies = stat_data
    
    histogram_data = []
    for f in forecast_accuracies:
        f: datetime.timedelta
        histogram_data.append(f.total_seconds())
    
    with open(results_file, 'w') as f:
        f.write(stats)
        f.write(str(histogram_data)[1:-1])
    
    
    #print(sim.getStatData())