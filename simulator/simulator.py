from typing import Dict, List, Tuple, NewType, Callable
from queue import PriorityQueue
import datetime
import abc
import warnings

DEBUG_PRINT = False
DEBUG_PRINT_TYPES = ['Statistics', 'ComputeNodes', 'LicenseManager', 'JobHandler', 'JobHandler_ComputeNodes']
#DEBUG_PRINT_TYPES = []
#DEBUG_PRINT_TYPES.remove('Statistics')
#DEBUG_PRINT_TYPES.remove('ComputeNodes')
#DEBUG_PRINT_TYPES.remove('LicenseManager')
#DEBUG_PRINT_TYPES.remove('JobHandler')

# define types
TimeType = NewType('TimeType', datetime.datetime)
DurationType = NewType('DurationType', datetime.timedelta)

JobDuration = DurationType
JobSubmission = TimeType
JobCompletion = TimeType

JobId = str
QueueName = str

LicenseName = str
LicenseCheckOut = DurationType
LicenseUseDuration = DurationType

LicenseType = NewType('LicenseType', Tuple[LicenseName, LicenseCheckOut, LicenseUseDuration])
uidType = int
class License:
    def __init__(self, license: LicenseType, uid: uidType):
        self._data = list(license)
        self._uid = uid
    
    def getName(self) -> LicenseName:
        return self._data[0]
    
    def getCheckOut(self) -> DurationType:
        return self._data[1]
    
    def getDuration(self) -> DurationType:
        return self._data[2]
    
    def getCheckIn(self) -> DurationType:
        return self.getCheckOut() + self.getDuration()
    
    def getUID(self) -> uidType:
        return self._uid

LicenseContainerType =  NewType('LicenseContainerType', List[License])
class LicenseContainer:
    def __init__(self, licenseContainer: LicenseContainerType):
        self._data = licenseContainer
    
    def getNumLicenses(self):
        return len(self._data)
    
    def getLicenseContainer(self) -> LicenseContainerType:
        return self._data
    
    def count(self, string: str) -> int:
        cnt = 0
        for license in self._data:
            license: License
            if license.getName() == string:
                cnt += 1
        return cnt

    def getLicenseNames(self) -> List[str]:
        l = []
        for license in self._data:
            l.append(license.getName())
        return l
        
HistoricQueueTime = DurationType
JobType = NewType('JobType', Tuple[JobId, JobDuration, LicenseContainer, QueueName, HistoricQueueTime])
class Job:
    # define this so that priority queue works
    def __gt__(self, other):
        return False

    # define this so that priority queue works
    def __lt__(self, other):
        return False
    
    def __init__(self, job: JobType):
        self._data = job
    
    def getDuration(self) -> JobDuration:
        return self._data[1]
    
    def getLicenses(self) -> LicenseContainer:
        return self._data[2]
    
    def getJobId(self) -> JobId:
        return self._data[0]
    
    def getQueueName(self) -> QueueName:
        return self._data[3]
    
    def getQueueDuration(self) -> HistoricQueueTime:
        return self._data[4]

class LicenseLog():
    '''
    keeps track of license usage for forecasting
    '''
    def __init__(self) -> None:
        pass
    
    def logCheckOut(self) -> None:
        pass
    
    def logCheckIn(self) -> None:
        pass
    
    def getForecastData(self) -> None:
        pass

class Statistics:
    def __init__(self, log_start: TimeType = None, log_end: TimeType = None) -> None:
        self._jobSubmission: Dict[JobId, Tuple(TimeType, int)] = {}
        self._jobKills: Dict[JobId, List[TimeType]] = {}
        self._jobKillsNoResub: Dict[JobId, List[TimeType]] = {}
        self._wastedWork: Dict[JobId, List[TimeType]] = {}
        self._jobCompletion: Dict[JobId, TimeType] = {}
        self._logStart: TimeType = log_start
        self._logEnd: TimeType = log_end
        self._current_forecasts: Dict[LicenseName, TimeType] = {}
        self._forecast_accuracies: List[DurationType] = []
        
        self._licenseDurations: Dict[LicenseName, Dict[JobId, List[Tuple[TimeType, uidType]]]] = {}
    
    def killJob(self, job_id: JobId, cur_time: TimeType, exec_start_time: TimeType) -> None:
        if self._jobSubmission.get(job_id) == None:
            raise Exception(f"Statistics Error: Job {job_id} killed before recording submission")
        
        if self._jobKills.get(job_id) == None:
            self._jobKills[job_id] = []
            self._wastedWork[job_id] = []
        
        self._jobKills[job_id].append(cur_time)
        self._wastedWork[job_id].append(cur_time-exec_start_time)
        
    def killJobNoResub(self, job_id: JobId, cur_time: TimeType, exec_start_time: TimeType) -> None:
        if self._jobSubmission.get(job_id) == None:
            raise Exception(f"Statistics Error: Job {job_id} killed with no resubmission before recording submission")
        
        self.killJob(job_id, cur_time, exec_start_time)
        
        if self._jobKillsNoResub.get(job_id) == None:
            self._jobKillsNoResub[job_id] = cur_time
        else:
            raise Exception(f"Statistics Error: Job {job_id} killed with no resubmission multiple times at time: {cur_time}")
    
    def queueJob(self, job_id: JobId, cur_time: TimeType, num_licenses: int):
        if self._jobSubmission.get(job_id) == None:
            has_licenses = (num_licenses>0)
            self._jobSubmission[job_id] = (cur_time, has_licenses)
        else:
            raise Exception(f"Statistics Error: Job {job_id} called queueJob() multiple times at time {cur_time} (first queue {self._jobKills[job_id]}). Are there duplicate Job IDs?")
    
    def finishJob(self, job_id: JobId, cur_time: TimeType):
        if self._jobCompletion.get(job_id) == None:
            self._jobCompletion[job_id] = cur_time
        else:
            raise Exception(f"Statistics Error: Job {job_id} called finishJob() multiple times at time {cur_time} (first completion {self._jobCompletion[job_id]})")
    
    def checkOut(self, licenses: List[LicenseName], job_id: JobId, cur_time: TimeType, uids: List[uidType]) -> None:
        assert(len(licenses) == len(uids))
        assert(len(uids) == len(set(uids)))
        for license, uid in zip(licenses, uids):
            if self._licenseDurations.get(license) == None:
                self._licenseDurations[license]= {}
            if self._licenseDurations[license].get(job_id) == None:
                self._licenseDurations[license][job_id] = []
            
            self._licenseDurations[license][job_id].append((cur_time, uid))
            #assert (len(self._licenseDurations[license][job_id]) < 100)
    
    def checkIn(self, licenses: List[LicenseName], job_id: JobId, cur_time: TimeType, uids: List[uidType]) -> None:
        #print(f"Statistics: checking in {len(licenses)} licenses {licenses}  ")
        assert(len(licenses) == len(uids))
        assert(len(uids) == len(set(uids)))
        for license, uid in zip(licenses, uids):
            self.evaluateForecast(license, cur_time)
            if self._licenseDurations.get(license) == None:
                raise Exception(f"Statistics Error: Job {job_id} called checkIn() for license {license} at time {cur_time}, but license is not tracked by Statistics")
            if self._licenseDurations[license].get(job_id) == None:
                raise Exception(f"Statistics Error: Job {job_id} called checkIn() for license {license} at time {cur_time}, but Job is not tracked by Statistics")
            if len(self._licenseDurations[license][job_id]) == 0:
                raise Exception(f"Statistics Error: No existing licenses tracked for Job {job_id}, license {license} at time {cur_time}") 
            if len(self._licenseDurations[license][job_id]) > 1 and len(set(self._licenseDurations[license][job_id])) != 1:
                cur_uids = [item[1] for item in self._licenseDurations[license][job_id]]
                if len(set(cur_uids)) < len(cur_uids):
                    print(self._licenseDurations[license][job_id])
                    print(f"Statistics Warning: Multiple distinct licenses with same uid tracked for Job {job_id}, license {license} at time {cur_time}. Ambiguous as to which one to remove (is uid not being assigned correctly?)")
            
            #print(f"Statistics: license durations before {len(self._licenseDurations[license][job_id])}")
            # if single license
            if len(self._licenseDurations[license][job_id]) == 1:
                l, uid_ = self._licenseDurations[license][job_id].pop()

            # multiple licenses, with different check out times
            else:
                l = [item for item in self._licenseDurations[license][job_id] if item[1] == uid]
                if len(l) == 0:
                    raise Exception(f"Statistics Error: No existing licenses tracked for Job {job_id}, license {license}, uid {uid} at time {cur_time}") 
                elif len(l) > 1:
                    raise Exception(f"Statistics Error: Duplicate uid for licenses tracked for Job {job_id}, license {license}, uid {uid} at time {cur_time}") 
                self._licenseDurations[license][job_id].remove(l[0])
                
            # clean up data structure
            if len(self._licenseDurations[license][job_id]) == 0:
                del self._licenseDurations[license][job_id]
                
            if len(self._licenseDurations[license]) == 0:
                del self._licenseDurations[license]
            
            '''
            if self._licenseDurations.get(license) and self._licenseDurations[license].get(job_id):
                print(f"Statistics: license durations after {len(self._licenseDurations[license][job_id])}")
            else:
                print(f"Statistics: license durations after 0")
            '''
    
    def forecast(self, license_name: LicenseName, forecast_time: TimeType) -> None:
        if self._current_forecasts.get(license_name) == None:
            self._current_forecasts[license_name] = []
        self._current_forecasts[license_name].append(forecast_time)

    def evaluateForecast(self, license_name: LicenseName, cur_time: TimeType) -> None:
        if self._current_forecasts.get(license_name) == None or len(self._current_forecasts[license_name]) == 0:
            return
        for forecast_time in self._current_forecasts[license_name]:
            error = forecast_time - cur_time
            self._forecast_accuracies.append(error)
        self._current_forecasts[license_name] = []

    def getLicenseUseTimes(self, license_name: LicenseName, cur_time: TimeType) -> List[DurationType]:
        license_durations = []
        
        if len(self._licenseDurations[license_name]) == 0:
            raise Exception(f"Statistics Error: License {license_name} is not in use, but usage time requested at time {cur_time}. Does this license have a max use of 0?") # May indicate problem with license denial condition

        #if license_name == 'ACOUSTICS' and len(self._licenseDurations[license_name]) > 2:
        #    raise Exception(f"Statistics Error: License {license_name} too many in use at time {cur_time}") # May indicate problem with license denial condition


        for job_id, check_out_times in self._licenseDurations[license_name].items():
            if len(check_out_times) == 0:
                raise Exception(f"Statistics Error: Job {job_id} has no license durations for license {license_name} at time {cur_time}") # did empty job not get deleted in self.checkIn()?

            for check_out_time, uid in check_out_times:
                license_durations.append(cur_time-check_out_time)
        
        assert(len(license_durations)>0)    
        
        return license_durations
    
    def getStats(self, print_stats=True):
        num_jobs_killed = 0
        num_license_jobs = 0
        completion_times = []
        killed_job_completion_times = []
        failed_jobs = []
        for job_id in self._jobSubmission.keys():
            uses_license = self._jobSubmission[job_id][1]
            if not uses_license:
                continue
            num_license_jobs += 1
            if self._jobCompletion.get(job_id):
                completion_time = self._jobCompletion[job_id] - self._jobSubmission[job_id][0]
                completion_times.append(completion_time)
                if job_id in self._jobKills.keys():
                    killed_job_completion_times.append(completion_time)
            else:
                failed_jobs.append(job_id)
        num_jobs_killed = len(self._jobKills)
        num_job_killed_no_resub = len(self._jobKillsNoResub)
        num_kills = 0
        for job_id in self._jobKills.keys():
            num_kills += len(self._jobKills[job_id])
        
        zero = datetime.timedelta(0)
        results = f"Total Execution Time: {max(self._jobCompletion.values()) - min(self._jobSubmission.values())[0]}\n"
        results += f"Total Number of Jobs Submitted: {len(self._jobSubmission.keys())}\n"
        results += f"Total Number of License-using Jobs Submitted: {num_license_jobs}\n"
        results += f"Total Number of Jobs Completed: {len(self._jobCompletion.keys())}\n"
        results += f"Total Number of License-using Jobs Completed: {len(completion_times)}\n"
        results += f"Total Number of Jobs Failed to Complete: {len(failed_jobs)}\n"
        results += f"Total Number of Jobs Killed: {num_jobs_killed}\n"
        results += f"Total Number of Jobs Killed with No Resubmission: {num_job_killed_no_resub}\n"
        results += f"Total Number of Job Kills (Denials): {num_kills}\n"
        results += f"Average Number of License-using Job Kills: {num_jobs_killed/num_license_jobs}\n"
        if num_jobs_killed > 0:
            results += f"Average Number of Denied License-using Job Kills: {num_kills/num_jobs_killed}\n"
        else:
            results += f"Average Number of Denied License-using Job Kills: None\n"            
        results += f"Total Wasted Work: {sum([sum(x, zero) for x in self._wastedWork.values()], zero)}\n"
        if (len(self._wastedWork) > 0):
            avg_wasted_work = sum([sum(x, zero) for x in self._wastedWork.values()], zero) / len(self._wastedWork)
        else:
            avg_wasted_work = 0
        results += f"Average Wasted Work per Denied License-using Jobs: {avg_wasted_work}\n"
        results += f"Total License-using Job Completion Time: {sum(completion_times, zero)}\n"
        results += f"Average License-using Job Completion Time: {sum(completion_times, zero)/len(completion_times)}\n"
        results += f"Total Denied License-using Job Completion Time: {sum(killed_job_completion_times, zero)}\n"
        if len(killed_job_completion_times) > 0:
            results += f"Average Denied License-using Job Completion Time: {sum(killed_job_completion_times, zero)/len(killed_job_completion_times)}\n"
        else:
            results += f"Average Denied License-using Job Completion Time: None\n"
            
        if print_stats:
            print(results)
        
        return results

    def getStatData(self):
        return (self._jobSubmission, self._jobCompletion, self._jobKills, self._wastedWork, self._forecast_accuracies)
    
class ForecastModel(abc.ABC):
    @abc.abstractmethod
    def forecast(self, license_name: LicenseName, cur_time: TimeType, max_delay: TimeType, min_license: int, window_size: Tuple[TimeType, TimeType]) -> DurationType:
        '''
        license_name: Name of license to forecast
        cur_time: start time of forecasting
        max_delay: maximum delay time for license
        min_license: return forecast time where minimum license is less than or equal to this amount
        window_size: size of min_license window, e.g. if window_size=(2min, 2min), return data point where at least previous 2 minutes and at least subsequent 2 minutes are less than or equal to min_license. NOTE: granularity depends on model used, if model has 1 minute granularity, using a window size of (10sec, 10sec) will round UP to 1 minute granularity
        
        returns: how long to delay license/job for
        '''
        pass
    
    def init_stats(self, stats: Statistics) -> None:
        '''
        Add statistics tracker
        '''
        self._stats = stats
            
class LicenseManager:
    def __init__(self, licenses: Dict[str, int]) -> None:
        '''
        licenses = Dict[key:licenseName, value:maxAmount]
        '''
        
        self._licenses: Dict[LicenseName, List[int, int]] = {} # dictionary of lists: key = 'license name', value = [in_use, maximum]
                            # keeps track of how many licenses are in use
        self._name = 'LicenseManager' # for debugging

        for name in licenses.keys():
            max_amount = licenses[name]
            self._licenses[name] = [0, max_amount]
        
    def checkOut(self, license_names: List[str]) -> Tuple[bool, List[LicenseName], List[int]]:
        assert(len(license_names)>0)
        if DEBUG_PRINT and self._name in DEBUG_PRINT_TYPES:
            print(f"licenseManager: checkOut licenses {license_names}")
            print(f"\t license values before: ", end='')
            for l in list(set(license_names)):
                print(f"{l}:{self._licenses[l][0]}/{self._licenses[l][1]} ", end='')
            print('')
        enough_licenses = True
        failed_licenses = []
        failed_license_inds = [] # for tracking uids TODO find cleaner way to do this
        
        # check if it is possible to check out all licenses in license_names
        for ind, l in enumerate(license_names):
            current_use, max_amount = self._licenses[l]
            assert(current_use <= max_amount)
            if current_use + license_names.count(l) > max_amount:
                enough_licenses = False
                failed_licenses.append(l)
                failed_license_inds.append(ind)
                
        if not enough_licenses:
            if DEBUG_PRINT and self._name in DEBUG_PRINT_TYPES: print(f"\t not enough licenses for {failed_licenses}")
            return False, failed_licenses, failed_license_inds
        else: # if possible, check out all licenses in license_names
            for l in license_names:
                self._licenses[l][0] += 1
            
            if DEBUG_PRINT and self._name in DEBUG_PRINT_TYPES:    
                print(f"\t license values after: ", end='')
                for l in list(set(license_names)):
                    print(f"{l}: {self._licenses[l][0]}/{self._licenses[l][1]} ", end='')
                print('')
            return True, [], []
                
    def checkIn(self, license_names: List[str]) -> None:
        if DEBUG_PRINT and self._name in DEBUG_PRINT_TYPES: print(f"licenseManager: checkIn licenses {license_names}")
        assert(type(license_names) == list)
        for l in license_names:
            if self._licenses[l][0] > 0:
                self._licenses[l][0] -= 1
            else:
                raise Exception(f"License {l} is not checked out {self._licenses[l]}")
    
    # TODO make more sophisticated, same licenses may check in before other of same checkout in a job (but unlikely)
    def checkIfPossible(self, license_container: LicenseContainer) -> bool:
        '''
        Check if all of these licenses could be checked out in best case
        '''
        license_names = license_container.getLicenseNames()
        for l in list(set(license_names)):
            current_use, max_amount = self._licenses[l]
            requested_amount = license_names.count(l)
            if requested_amount > max_amount:
                return False
        return True
    
class JobQueue:
    def __init__(self, queues: List[str]) -> None:
        '''
        queues should be in decreasing order of priority
        '''
        self._queues = {}
        
        for q in queues:
            self._queues[q] = []
    
    def addJob(self, job: Job, queue: str) -> None:
        assert(job != None)
        self._queues[queue].append(job)
        
    def getJobQ(self, queue: str) -> Job:
        return self._queues[queue].pop(0)
        
    # get job by queue priority
    def getJob(self) -> Job:
        for q in self.getQueueNames():
            if not self.isEmptyQ(q):
                return self.getJobQ(q)
        
    def isEmptyQ(self, queue: str) -> bool:
        return (len(self._queues[queue]) == 0)
    
    def isEmpty(self) -> bool:
        for q in self.getQueueNames():
            if not self.isEmptyQ(q):
                return False
        return True
    
    def getQueueNames(self) -> List[str]:
        return self._queues.keys()
    
    
class ComputeNodes:
    def __init__(self, license_manager: LicenseManager, max_compute_nodes: int, stats: Statistics) -> None:
        self._license_manager = license_manager
        self._max_compute_nodes = max_compute_nodes
        self._exec_jobs: Dict[JobId, Tuple(TimeType, Job)] = {} # jobs being executed key: jobid, value: (exec_start_time, Job)
        self._event_queue = PriorityQueue() # keeps events, so that time in between can be skipped
        self._stats = stats
        self._name = 'ComputeNodes' # for debugging
        
    def numFreeNodes(self) -> int:
        return self._max_compute_nodes-len(self._exec_jobs)
    
    def exec(self, cur_time) -> List[Tuple[Job, TimeType, List[LicenseName]]]:
        
        killed_job_list = [] # jobs terminated that should be resubmitted
        no_resub_job_list = [] # jobs termineated that should NOT be resubmitted
        
        for key in list(self._exec_jobs.keys()):
            j = self._exec_jobs[key]
            start_exec_time = j[1] # refer to addJob for data layout
            job: Job = j[0]
            job_duration = job.getDuration()
            ###if DEBUG_PRINT  and self._name in DEBUG_PRINT_TYPES: print(f"computeNode: executing job {job.getJobId()}, duration {job_duration}, estimated completion {start_exec_time+job_duration}")
            num_licenses = job.getLicenses().getNumLicenses()

            ## CheckIn Licenses
            if num_licenses > 0:
                for license in job.getLicenses().getLicenseContainer():
                    if start_exec_time + license.getCheckIn() == cur_time:
                        self._license_manager.checkIn([license.getName()])
                        self._stats.checkIn([license.getName()], job.getJobId(), cur_time, [license.getUID()])
            ## CheckOut Licenses
            if num_licenses > 0:
                license_names_to_check_out = []
                license_uids_to_check_out = []
                licenses_to_check_out: List[License] = []
                for license in job.getLicenses().getLicenseContainer():
                    # put in list to deal with case of multiple licenses checkedOut in same second
                    if start_exec_time + license.getCheckOut() == cur_time:
                        license_names_to_check_out.append(license.getName())
                        license_uids_to_check_out.append(license.getUID())
                        licenses_to_check_out.append(license)
                if len(license_names_to_check_out)>0:        
                    # checking out ALL licenses CURRENTLY being requested at cur_time, for a SINGLE job
                    success, failed_licenses, failed_license_inds = self._license_manager.checkOut(license_names_to_check_out)
                    # TODO find cleaner way to do this
                    if success:
                        failed_uids = []
                        for ind in failed_license_inds:
                            failed_uids.append(license_uids_to_check_out[ind])
                        for uid in failed_uids:
                            license_uids_to_check_out.remove(uid)
                        stats_license_check_out = license_names_to_check_out.copy()
                        for l in failed_licenses: 
                            stats_license_check_out.remove(l)
                        self._stats.checkOut(stats_license_check_out, job.getJobId(), cur_time, license_uids_to_check_out)
                    # check if successfully aquired license(s)
                    if not success:
                        # check of job possible to run with max licenses
                        #is_possible = self._license_manager.checkIfPossible(license_names_to_check_out)
                        is_possible = self._license_manager.checkIfPossible(job.getLicenses())
                        
                        if is_possible:
                            if DEBUG_PRINT and self._name in DEBUG_PRINT_TYPES: print(f"computeNode: killed job {job.getJobId()} at {cur_time}, job_exec_start {start_exec_time}")
                            self.killJob(job, start_exec_time, cur_time)
                            killed_job_list.append((job, start_exec_time, failed_licenses))
                        else:
                            if DEBUG_PRINT and self._name in DEBUG_PRINT_TYPES: print(f"computeNode: killed job NO RESUBMISSION {job.getJobId()} at {cur_time}, job_exec_start {start_exec_time}")
                            self.killJob(job, start_exec_time, cur_time)
                            no_resub_job_list.append((job, start_exec_time, failed_licenses))
                    else: # if success, checkIn licenses from License Manager and add event for checkOuts
                        for license in licenses_to_check_out:
                            check_in_event = cur_time + license.getDuration()
                            if check_in_event > cur_time:
                                self._event_queue.put((check_in_event, "checkIn"))
                            else:
                                assert(check_in_event == cur_time)
                                print(f"WARNING, job {job.getJobId()} license {license.getName()} has 0 use duration")
                                self._license_manager.checkIn([license.getName()])
                                self._stats.checkIn([license.getName()], job.getJobId(), cur_time, [license.getUID()])

            ## Finish job
            if start_exec_time+job_duration <= cur_time:
                if DEBUG_PRINT and self._name in DEBUG_PRINT_TYPES: print(f"computeNode: Finished job {job.getJobId()} at {cur_time}, job_exec_start {start_exec_time}")
                if start_exec_time+job_duration < cur_time:
                    print(f"WARNING, job {job.getJobId()} removed {start_exec_time+job_duration-cur_time}s after execution finished at time {cur_time}")
                # check if all licenses are checkedIn
                if num_licenses != 0:
                    for license in job.getLicenses().getLicenseContainer():
                        if start_exec_time + license.getCheckOut() + license.getDuration() > cur_time:
                            print(f"WARNING, license {license.getName()} not checkedIn during job {job.getJobId()} end")
                            self._license_manager.checkIn([license.getName()])
                            self._stats.checkIn([license.getName()], job.getJobId(), cur_time, [license.getUID()])
                # remove job from execution node
                self._exec_jobs.pop(job.getJobId())
                self._stats.finishJob(job.getJobId(), cur_time)
                
                
        return killed_job_list, no_resub_job_list
                            

    def killJob(self, job: Job, start_exec_time: TimeType, cur_time: TimeType) -> None:
        #remove all active licenses in job
        job_id = job.getJobId()
        num_licenses = job.getLicenses().getNumLicenses()
        
        licenses_to_check_in = []
        licence_uids_to_check_in = []
        for license in job.getLicenses().getLicenseContainer():
            # find jobs that were checkedOut but not checkedIn
            if start_exec_time + license.getCheckOut() < cur_time and start_exec_time + license.getCheckIn() > cur_time: # licenses are first checkedIn ar cur_time, then attempted to checkOut
                licenses_to_check_in.append(license.getName())
                licence_uids_to_check_in.append(license.getUID())
        
        if len(licenses_to_check_in) > 0:
            self._license_manager.checkIn(licenses_to_check_in)
            self._stats.checkIn(licenses_to_check_in, job_id, cur_time, licence_uids_to_check_in)
        
        # remove job from execution node
        self._exec_jobs.pop(job_id)
        
    
    def addJob(self, job: Job, cur_time: TimeType) -> None:
        assert(job != None)
        if len(self._exec_jobs) >= self._max_compute_nodes:
            raise Exception(f"Not enough compute nodes for job {job[0]}")
        else:
            assert (self._exec_jobs.get(job.getJobId()) == None)
            self._exec_jobs[job.getJobId()] = (job, cur_time)  
            # add future events (job end, license checkOut, license checkIn) to event queue
            assert(cur_time + job.getDuration() > cur_time)
            self._event_queue.put((cur_time + job.getDuration(), "jobEnd"))
            if job.getLicenses().getNumLicenses() > 0:
                for license in job.getLicenses().getLicenseContainer():
                    check_out_event = cur_time + license.getCheckOut()
                    if check_out_event > cur_time:
                        self._event_queue.put((check_out_event, "checkOut"))
                    # put checkIn event after successfully checked out
                    #check_in_event = cur_time + license.getCheckIn()
                    #if check_in_event > cur_time:
                    #    self._event_queue.put((check_in_event, "checkIn"))
                    
    def hasEvent(self) -> bool:
        return not self._event_queue.empty()                
                    
    def viewNextEvent(self) -> Tuple[int, str]:
        return self._event_queue.queue[0]
    
    def removeNextEvent(self) -> None:
        self._event_queue.get()
    
# TODO ADD RESUBMISSION LIMIT-> USE DICT TO KEEP TRACK
class JobHandler:
    def __init__(self, start_time: TimeType, job_queue: JobQueue, compute_nodes: ComputeNodes, stats: Statistics, resubmission_delay: DurationType = 0, forecast_model: ForecastModel = None, forecast_parameters: Tuple[TimeType, float, Tuple[TimeType, TimeType]] = None, use_historic_queue_time = False) -> None:
        '''
        forecast_parameters = (max_delay, min_license_percentage, (window_size_left, window_size_right)))
        '''
        self._resubmission_queue: PriorityQueue[TimeType, Job] = PriorityQueue() # key=cur_time+delay value=delay time, job, queue
        self._job_queue = job_queue
        self._compute_nodes = compute_nodes
        self._stats = stats
        self._resubmission_delay = resubmission_delay
        self._forecast_model = forecast_model
        self._forecast_parameters = forecast_parameters
        self._max_licence_amount = compute_nodes._license_manager._licenses
        self._last_job_submission_time = None # if added a job, but did not run. submission time of job is an event
        self._use_historic_queue_time = use_historic_queue_time
        self._name = 'JobHandler' # for debugging
        if forecast_model != None:
            if forecast_parameters == None:
                raise Exception("Error, must have forecast_parameters if using forecasting")
    
    def queueJob(self, job: Job, queue_name: str, cur_time: TimeType) -> None:
        self._job_queue.addJob(job, queue_name)
        # Statistics Tracking
        if self._use_historic_queue_time:
            # !!IMPORTANT NOTE: Assuming job did not wait in JobQueue, i.e. unlimited resources
            self._stats.queueJob(job.getJobId(), cur_time-job.getQueueDuration(), job.getLicenses().getNumLicenses())
        else:
            self._stats.queueJob(job.getJobId(), cur_time, job.getLicenses().getNumLicenses())
        
        if self._compute_nodes.numFreeNodes() > 0: # can start running job immediately, so add an event
            self._last_job_submission_time = cur_time
        ##self.run(cur_time)
    
    def run(self, cur_time: TimeType) -> None:
        ## add jobs in resubmission queue to job queue after delay
        #self._last_job_submission_time = None
        if DEBUG_PRINT and self._name+'_ComputeNodes' in DEBUG_PRINT_TYPES: print(f"jobHandler: number of free compute nodes {self._compute_nodes.numFreeNodes()}")
        while (not self._resubmission_queue.empty()) and self._resubmission_queue.queue[0][0] <= cur_time:
            job: Job
            delayed_submission_time, job = self._resubmission_queue.get()
            if delayed_submission_time < cur_time:
                print(f"WARNING, job {job.getJobId()} in resubmission queue waited longer than delay {delayed_submission_time} at time {cur_time}")
            self._job_queue.addJob(job, job.getQueueName())
        
        ## add jobs from queue if compute node available
        while self._compute_nodes.numFreeNodes() > 0 and not self._job_queue.isEmpty():
            job_to_add = self._job_queue.getJob()
            if DEBUG_PRINT and self._name in DEBUG_PRINT_TYPES: print(f"jobHandler: adding job {job_to_add.getJobId()}")
            self._compute_nodes.addJob(job_to_add, cur_time)
            
        ## try executing jobs
        if DEBUG_PRINT and self._name+'_ComputeNodes' in DEBUG_PRINT_TYPES: print(f"jobHandler: executing compute nodes, time {cur_time}")
        failed_jobs, no_resub_jobs = self._compute_nodes.exec(cur_time)
        
        ## resubmit any failed jobs
        if len(failed_jobs)+len(no_resub_jobs) > 0:
            # add failed jobs to resubmission queue
            for job, exec_start_time, denied_licenses in failed_jobs:
                # track statistics call
                self._stats.killJob(job.getJobId(), cur_time, exec_start_time)

                if self._forecast_model == None:
                    if self._use_historic_queue_time:
                        resub_delay = cur_time+job.getQueueDuration()
                    else:
                        resub_delay = cur_time+self._resubmission_delay                     
                    self._resubmission_queue.put((resub_delay, job))
                else:
                    # Use forecasting model to add delay. Note: delay cannot be smaller can self._resubmission_delay
                    
                    # TODO finish these
                    if len(denied_licenses) > 1:
                        print(f"Warning: forecasting multiple denied licenses not implemented, using first denied license {denied_licenses[0]}. Denied Licenses: {denied_licenses}")
                    if len(denied_licenses) > len(list(set(denied_licenses))):
                        print(f"Warning: forecasting multiple of SAME denied licenses not implemented")
                        
                    denied_license_name = denied_licenses[0]
                    
                    max_delay, min_license_percentage, window_size = self._forecast_parameters
                    max_license_amount = self._max_licence_amount[denied_license_name][1]
                    min_license = int(max_license_amount*min_license_percentage) # take FLOOR of result
                    
                    forecast_delay = self._forecast_model.forecast(denied_license_name, cur_time, max_delay, min_license, window_size, job_id = job.getJobId())
                    print(f"jobHandler: Forecast delay for job {job.getJobId()} = {forecast_delay}")
                    if self._use_historic_queue_time:
                        delayed_submission_time = max(forecast_delay, job.getQueueDuration())
                    else:
                        delayed_submission_time = max(forecast_delay, self._resubmission_delay)
                        
                    self._resubmission_queue.put((cur_time+delayed_submission_time, job)) 
                    self._stats.forecast(denied_license_name, cur_time+delayed_submission_time)
            for job, exec_start_time, denied_licenses in no_resub_jobs:
                # track statistics call
                self._stats.killJobNoResub(job.getJobId(), cur_time, exec_start_time)
    
    # double check this TODO
    def hasNextEvent(self) -> bool:
        return (self._last_job_submission_time != None) or self._compute_nodes.hasEvent() or (not self._resubmission_queue.empty())
    
    def viewNextEvent(self) -> TimeType:
        '''
        Returns next event without removing it from its event queue
        '''     
        compute_event = None
        resubmission_event = None 
        events = []          
        if self._compute_nodes.hasEvent():
            events.append(self._compute_nodes.viewNextEvent()[0])
        if not self._resubmission_queue.empty():
            events.append(self._resubmission_queue.queue[0][0])
        if not self._last_job_submission_time is None:
            events.append(self._last_job_submission_time)
        
        assert(len(events) != 0)

        return min(events)

                    
    def removeNextEvent(self) -> TimeType:
        '''
        Returns next event AND removes it from its event queue along with other events with the same time
        '''                            

        events = []       
        if self._compute_nodes.hasEvent():
            events.append(self._compute_nodes.viewNextEvent()[0]) # add key
        if not self._resubmission_queue.empty():
            events.append(self._resubmission_queue.queue[0][0])
        if not self._last_job_submission_time is None:
            events.append(self._last_job_submission_time)
        
        assert(len(events) > 0)            

        next_event = min(events)
        
        # remove all events with same time
        while self._compute_nodes.hasEvent() and next_event == self._compute_nodes.viewNextEvent()[0]:
            self._compute_nodes.removeNextEvent()
        
        if not self._last_job_submission_time is None:
            if self._last_job_submission_time == next_event:
                self._last_job_submission_time = None
            else:
                raise Exception(f"Error, invalid control flow. Next event {next_event}, last_job_submission_time {self._last_job_submission_time}")
        
        # Do not remove events from resubmission queue, since all resubmissions with same resubmission time will be submitted at once during the respective time in self.run() and removed from event queue there
        
        return next_event
        
        '''
        if resubmission_event == None or compute_event < resubmission_event:
            self._compute_nodes.removeNextEvent()
            while self._compute_nodes.hasEvent() and self._compute_nodes.viewNextEvent()[0] <= compute_event: # remove all events with same time
                assert(self._compute_nodes.viewNextEvent()[0] == compute_event) # sanity check
                self._compute_nodes.removeNextEvent()
            return compute_event
        
        elif compute_event == None or resubmission_event < compute_event: # all resubmissions with same resubmission time will be submitted at once during the respective time in self.run()
            return resubmission_event
        
        elif resubmission_event == compute_event:
            self._compute_nodes.removeNextEvent()
            while self._compute_nodes.viewNextEvent()[0] <= compute_event: # remove all events with same time
                assert(self._compute_nodes.viewNextEvent()[0] == compute_event) # sanity check
                self._compute_nodes.removeNextEvent()
            return compute_event
        
        else:
            raise Exception("Error, invalid control flow")
        '''
        
    def printAllEvents(self): # for debugging purposes
        print('Compute node event queue:')
        for e in self._compute_nodes._event_queue.queue:
            print(f"{e[0]} {e[1]}")
        print('resubmission queue event queue:')
        for e in self._resubmission_queue.queue:
            print(e)

class JobStream():
    '''
    Reads job data and passes Jobs to Simulator
    '''
    
    def __init__(self, file_name: str = None, queue_names: List[str] = [], job_data: List[Tuple[TimeType, Job]] = [], sort = False, toTimeType: Callable[[str], TimeType] = None, toDurationType: Callable[[str], DurationType] = None, use_historic_queue_time = False, ignored_licenses: List[LicenseName] = [], max_job_duration: DurationType = None, split_long_jobs = True) -> None:
        '''
        Format of each line in file: jobId, queueTime, jobDuration, queueName; licenseName, checkOut(time in sec after job starts), licenseDuration; licenseName, checkOut, licenseDuration; ...
        
        max_job_duration filters jobs over max size
        '''
        if file_name != None:
            self._job_data: List[Tuple[TimeType, Job]] = []
            uid = 0 # unique id for each license
            with open(file_name, 'r') as f:
                for line in f.readlines():
                    _ = line.strip().split(';')
                    hist_queue_duration = None
                    if use_historic_queue_time:
                        #print(_)
                        jobId, queue_time, jobDuration, queue_name, hist_queue_duration = _.pop(0).split(',')
                        hist_queue_duration = toDurationType(hist_queue_duration)
                    else:
                        jobId, queue_time, jobDuration, queue_name = _.pop(0).split(',')
                    if not queue_name in queue_names: # if not in queues, add to lowest priority queue
                        queue_name = queue_names[-1]

                    queue_time = toTimeType(queue_time)
                    if use_historic_queue_time:
                        queue_time += hist_queue_duration # submit job when 'queueing' is finished

                    jobDuration = toDurationType(jobDuration)
                    assert(jobDuration > toDurationType('0'))
                    license_container: LicenseContainerType = []

                    for _i in _:
                        i = _i.strip().split(',')
                        license_name = i[0]
                        if license_name in ignored_licenses:
                            continue
                        check_out = toDurationType(i[1])
                        license_duration = toDurationType(i[2])
                        license_container.append(License((license_name, check_out, license_duration), uid))
                        uid += 1
                    if use_historic_queue_time:
                        job = Job((jobId, jobDuration, LicenseContainer(license_container), queue_name, hist_queue_duration))
                        if job.getLicenses().getNumLicenses() == 0:
                            continue
                    else:
                        job = Job((jobId, jobDuration, LicenseContainer(license_container), queue_name, None))
                
                    if not self._check_if_valid_job(job):
                        print(f"Warning job file contains invalid jobs: jobid {job.getJobId()}")
                        continue
                    # filter jobs that are too long
                    if max_job_duration != None and jobDuration > max_job_duration:
                        print(f"JobStream: Skipping job {jobId} of duration {jobDuration} > max_job_duration {max_job_duration}")
                        if split_long_jobs: # TODO this is a temporary workaround, implement this in job_parse.py
                            split_job_list, queue_times_list = self._split_job(job, max_job_duration=max_job_duration, queue_name=queue_name, job_queue_time=queue_time, hist_queue_duration=hist_queue_duration)
                            for split_job, queue_time_ in zip(split_job_list, queue_times_list):
                                self._job_data.append((queue_time_, split_job))
                        else:
                            continue
                    else:
                        self._job_data.append((queue_time, job))
            self._job_data.sort(key=lambda x: x[0])
        else:
            assert(job_data != None)
            self._job_data = job_data
            if sort:
                self._job_data.sort(key=lambda x: x[0])
    
    def _check_if_valid_job(self, job: Job) -> bool:
        zero_duration = datetime.timedelta(seconds=0)
        job_duration = job.getDuration()

        for license in job.getLicenses().getLicenseContainer():
            if license.getCheckOut()<zero_duration or license.getCheckIn()<zero_duration or license.getDuration()<zero_duration:
                print(f"Negative license duration for jobid {job.getJobId()}")
                return False
            if license.getCheckOut()>job_duration or license.getCheckIn()>job_duration or license.getDuration()>job_duration:
                if license.getCheckOut()>job_duration:
                    print(f"Too large license checkOut {license.getCheckOut()} > {job_duration} for jobid {job.getJobId()}")
                if license.getCheckIn()>job_duration:
                    print(f"Too large license checkIn {license.getCheckIn()} > {job_duration} for jobid {job.getJobId()}")
                if license.getDuration()>job_duration:
                    print(f"Too large license duration {license.getDuration()} > {job_duration} for jobid {job.getJobId()}")
                return False
            if license.getCheckIn() - license.getCheckOut() == 0 or license.getDuration() == 0:
                print(f"License 0 duration")
                return False
        return True
            
            

            
    
    def _split_job(self, job: Job, max_job_duration: DurationType, queue_name: QueueName, job_queue_time: TimeType, hist_queue_duration: DurationType = None) -> Tuple[List[Job], List[TimeType]]:
        '''
        Convert long jobs into smaller sized jobs
        
        TODO this should be done in job_parse.py
        '''
        
        job_start_delay: DurationType = datetime.timedelta(seconds=40) # how long it takes for job to request first license 
        split_jobs_list: List[Job] = [] # will be returned
        queue_time_list: List[TimeType] = [] # will be returned
        split_id = 0
        licenses_list: List[License] = job.getLicenses().getLicenseContainer()
        licenses_list.sort(key=lambda x: x.getCheckOut())
        jobid = job.getJobId()
        
        
        cur_time = job_queue_time 
        new_job_licenses: List[License] = []
        max_checkIn_time: TimeType = datetime.datetime.min
        for ind, license in enumerate(licenses_list):
            #print(f"job queue time {job_queue_time}, job duration {job.getDuration()}, jobid {job.getJobId()}, license checkout duration {license.getCheckOut()}, license duration {license.getDuration()}, checkin {license.getCheckIn()}")
            
            if cur_time == None:
                cur_time = job_queue_time + license.getCheckOut() - job_start_delay
            if license.getDuration() > max_job_duration:
                print(f"JobID {jobid}, skipping License {license.getName()}: duration {license.getDuration()} > max job duration {max_job_duration}")
                continue
            prev_duration = license.getDuration()
            # update check out duration from job start
            license._data[1] -= cur_time-job_queue_time
            
            # sanity check
            #print(f"ind {ind}, split id {split_id}")
            assert(license.getCheckIn() == license.getCheckOut()+prev_duration)
            
            new_job_licenses.append(license)
            min_checkOut_time = job_queue_time + new_job_licenses[0].getCheckOut()
            
            min_checkOut_time_ = datetime.datetime.max
            for l in new_job_licenses:
                min_checkOut_time_ = min(min_checkOut_time_, job_queue_time+l.getCheckOut())
            assert(min_checkOut_time_ == min_checkOut_time)
            #print(f"job queue time {job_queue_time}, type {type(job_queue_time)}")
            #print(f"max checkin time {max_checkIn_time}, type {type(max_checkIn_time)}")
            #print(f"check out time {job_queue_time + license.getCheckOut()}, type {type(job_queue_time + license.getCheckOut())}")
            
            #max_checkIn_time = max(max_checkIn_time, job_queue_time+license.getCheckOut())
            max_checkIn_time = datetime.datetime.min
            for l in new_job_licenses:
                max_checkIn_time = max(max_checkIn_time, job_queue_time + l.getCheckIn())
            
            
            #print(f"max checkin {max_checkIn_time}, min checkout {min_checkOut_time}, differece {max_checkIn_time-min_checkOut_time}")
            # check if next license will cause job to be too long
            if ind+1 < len(licenses_list):
                duration_with_next_license = max(max_checkIn_time, job_queue_time + licenses_list[ind+1].getCheckIn()) - min_checkOut_time
            
            
            if ind+1 == len(licenses_list) or duration_with_next_license > max_job_duration:
                # construct new job with current licenses
                job_duration = max_checkIn_time-min_checkOut_time + (min_checkOut_time - cur_time)
                assert(max_checkIn_time-min_checkOut_time <= max_job_duration)
                new_jobid = f"{jobid}_split_{split_id}"
                split_id += 1
                
                new_job = Job((new_jobid, job_duration, LicenseContainer(new_job_licenses.copy()), queue_name, hist_queue_duration))
                new_job_queue_time = min_checkOut_time - job_start_delay
                
                assert(self._check_if_valid_job(new_job))
                
                split_jobs_list.append(new_job)
                queue_time_list.append(new_job_queue_time)
                
                # reset new_job_licenses list, and add previously omitted license
                new_job_licenses = []


        return split_jobs_list, queue_time_list
 
        # make any remaining licenses into a job
        if len(new_job_licenses) > 0:
            min_checkOut_time = job_queue_time + new_job_licenses[0].getCheckOut()
            
            min_checkOut_time_ = datetime.datetime.max
            for l in new_job_licenses:
                min_checkOut_time_ = min(min_checkOut_time_, job_queue_time+l.getCheckOut())
            assert(min_checkOut_time_ == min_checkOut_time)
            
            max_checkIn_time = datetime.datetime.min
            for l in new_job_licenses:
                max_checkIn_time = max(max_checkIn_time, job_queue_time + l.getCheckIn())
            
            # check if next license will cause job to be too long
            remaining_license_duration = max_checkIn_time - min_checkOut_time
            assert(remaining_license_duration <= max_job_duration)
            
            # construct new job with current licenses
            job_duration = max_checkIn_time-min_checkOut_time + job_start_delay

            new_jobid = f"{jobid}_split_{split_id}"
            #split_id += 1
            
            new_job = Job((new_jobid, job_duration, LicenseContainer(new_job_licenses.copy()), queue_name, hist_queue_duration))
            new_job_queue_time = min_checkOut_time - job_start_delay
            
            assert(self._check_if_valid_job(new_job))

            split_jobs_list.append(new_job)
            queue_time_list.append(new_job_queue_time)

        return split_jobs_list, queue_time_list
        
    
    def nextArrivalTime(self) -> TimeType:
        return self._job_data[0][0]
    
    def jobsLeft(self) -> int:
        return len(self._job_data)
    
    def getNextJob(self) -> Job:
        return self._job_data.pop(0)[1]


class Simulator():
    def __init__(self, queue_names: List[str], licenses: Dict[str, int], max_compute_nodes: int, job_stream: JobStream, resubmission_delay: DurationType = 0,  start_time: TimeType = 0, forecast_model: ForecastModel = None, forecast_parameters: Tuple[TimeType, float, Tuple[TimeType, TimeType]] = None, use_historic_queue_time = False) -> None:
        self._job_stream = job_stream
        self._cur_time = start_time
        self._stats = Statistics()
        if forecast_model != None:
            forecast_model.init_stats(self._stats)
        self._job_handler = JobHandler(start_time, 
                                       JobQueue(queue_names), 
                                       ComputeNodes(LicenseManager(licenses), max_compute_nodes, self._stats), 
                                       self._stats, 
                                       resubmission_delay,
                                       forecast_model,
                                       forecast_parameters,
                                       use_historic_queue_time)
        self._name = 'Simulator' # for debugging
    
    def run(self, end_time=None) -> None:
        run_until_end = False
        if end_time == None:
            run_until_end = True
        number_of_exec = 0
        prev_time = None
        while run_until_end or self._cur_time <= end_time:
            if DEBUG_PRINT and self._name in DEBUG_PRINT_TYPES: print('----')
            #input()
            #self._job_handler.printAllEvents() # for debugging purposes
            # add arriving jobs if no event occurring or job arrival occurs before event
            while (self._job_stream.jobsLeft() > 0) and ((not self._job_handler.hasNextEvent()) or self._job_stream.nextArrivalTime() <= self._job_handler.viewNextEvent()):
                assert(self._cur_time==0 or self._cur_time <= self._job_stream.nextArrivalTime())
                self._cur_time = self._job_stream.nextArrivalTime()
                next_job = self._job_stream.getNextJob()
                if DEBUG_PRINT and self._name in DEBUG_PRINT_TYPES: print(f'simulator: adding job {next_job.getJobId()} {self._cur_time}')
                self._job_handler.queueJob(next_job, next_job.getQueueName(), self._cur_time)
                
                # if added job to job handler, need to 
                #if self._job_handler.hasNextEvent() == False:
                #    break
            

            assert(not self._job_handler.hasNextEvent() or self._cur_time <= self._job_handler.viewNextEvent()) # sanity check
            
            # get next event time
            if self._job_handler.hasNextEvent():
                next_event_time = self._job_handler.viewNextEvent()
                if DEBUG_PRINT and self._name in DEBUG_PRINT_TYPES: print(f"simulator: next event {next_event_time}")
                self._job_handler.removeNextEvent()
                assert((prev_time is None) or prev_time < next_event_time)
                # run events at current time
                self._job_handler.run(next_event_time)
                
                self._cur_time = next_event_time
            else:
                # no events remaining
                print(f"no events left, time {self._cur_time}")
                break
            number_of_exec += 1
            prev_time = self._cur_time
        print(f"simulator: total number of event executions {number_of_exec}")
    def getStats(self, print_stats=True):
        return self._stats.getStats()
    
    def getStatData(self) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        return self._stats.getStatData()
    
if __name__ == "__main__":
    pass