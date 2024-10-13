from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import numpy
import math
import utils
import models
import csv
import time
import sys
import random
import copy

import flags
FLAGS = flags.FLAGS

from runtime.rpc_stubs.master_to_worker_pb2 import JobInfo

class _TFJobs(object):

    
    class g_job(object):
        def __init__(self, cpu, total_cpu=0):
            self.num_cpu = num_cpu       
            self.name = str(num_cpu) + '-cpu'
            self.total_job = 0
            self.end_job = 0
            self.num_queue = 2
            self.queues = [list() for i in range(self.num_queue)]
            self.queue_limit = [3600, 7200, 18000]
            self.total_cpu = total_cpu
            self.free_cpu = total_cpu
            self.running_jobs = list()
            self.pending_jobs = list()
            self.runnable_jobs = list()

        def alloc_free_cpus(self, need_num):
            if self.free_cpu >= need_num:
                self.free_cpu -= need_num
                return True
            else:
                return False

        def release_job_cpu(self, num_job=1):
            if num_job < 0:
                utils.print_fn("Error: num_job < 0")
                exit()
            self.free_cpu += int(self.num_cpu * num_job)

        def empty_cpu_alloc(self):
            self.free_cpu = self.total_cpu

        def get_cpu_reservation(self, reserved_num):
            
            used = self.total_cpu - self.free_cpu
            self.total_cpu = reserved_num
            self.free_cpu = self.total_cpu - used


        def get_cpu_demands(self):
            # return int((len(self.running_jobs) + len(self.pending_jobs)) * self.num_cpu)
            return int(len(self.runnable_jobs) * self.num_cpu)

    def __init__(self):
        self.num_job = 0        
        self.job_list = list()
        ''' job events is a list of tuple
            (time, dict)
        dict:
            'start_jobs': [xxx,xxx,xxx]
            'end_jobs': [xxx,xxx,xxx]
        '''
        self.job_events = list()        
        #holding pending jobs, add job_idx
        self.pending_jobs = list() # [{job_dict}, {job_dict}]
        self.runnable_jobs = list() # pending + running
        self.running_jobs = list() # running
        self.completed_jobs = list()

        self.migratable_jobs = list()
        self.num_queue = 3
        self.queues = [list() for i in range(self.num_queue)]
        self.queue_limit = [3250, 7200, 18000]

        # mem info in GB
        self.worker_mem = 5
        self.ps_mem = 6
        self.p_w_mem = 0.1

        self.cpu_job = dict()

        #gittins static delta
        self.gittins_delta = 3250

        self.mean_duration = 800
        self.job_dist_data = None

        self.overhead_list = []
        for _ in range(0, 5):
            self.overhead_list.append([])

    def get_job_model(self, job_dict):
        # if job_dict.has_key('model_name') and job_dict.has_key('model_scale'):
        if ('model_name' in job_dict) and ('model_scale' in job_dict):
            job_dict['model'] = models.get_model_with_scale(job_dict['model_name'], job_dict['model_scale'])
        else:
            utils.print_fn('Not enough model information to get the details')


    def get_network_load(self, job_dict):
        if 'num_cpu' not in job_dict:
            utils.print_fn('No cpu information')
            return 

        if 'model' not in job_dict:
            utils.print_fn('No model information')
            return
        
        num_w = job_dict['num_cpu']
        num_ps = num_w


        if num_w == 1:
            job_dict['ps_network'] = list()
            job_dict['w_network'] = list([0])

            '''
            check job ps_size 
            '''
            job_dict['ps_ave'] = 0
            return

        job_dict['w_network'] = list([job_dict['model']['total_size']] * num_w)
        job_dict['ps_network'] = list([0] * num_ps)
        for i in range(0, len(job_dict['model']['tensors'])):
            ps_idx = int(i % num_ps)
            # job_dict['ps_network'][ps_idx] += (job_dict['model']['tensors'][i] * num_w)
            job_dict['ps_network'][ps_idx] += (job_dict['model']['tensors'][i])

        for i in range(0, len(job_dict['ps_network'])):
            job_dict['ps_network'][i] = round(job_dict['ps_network'][i], 1)



    def add_job(self, job_dict):
        job_dict['resource_time'] = list()
        for key, value in job_dict.items():
        # for key, value in job_dict.iteritems():
            if (value is None) or ('resource_time' == key):
                continue
            if 'resource_time' in key:
                job_dict['resource_time'].append(float(value))
            elif value.isdigit():
                job_dict[key] = int(value)
        job_dict['duration'] = int(float(job_dict['duration']))
        
        job_dict['rank'] = sys.maxsize

        job_dict['submit_time'] /= 1000
        job_dict['duration'] /= 1000
        job_dict['iteration_time'] /= 1000
        job_dict['tput']  = 1/job_dict['iteration_time']
    

        if 'start_time' not in job_dict:
            job_dict['start_time'] = 0
        if 'end_time' not in job_dict:
            job_dict['end_time'] = 0
        if 'pending_time' not in job_dict:
            job_dict['pending_time'] = 0
        if 'multi-resource' in FLAGS.schedule or 'antman' in FLAGS.schedule:
            # job_dict['executed_iteration'] = 0.0
            job_dict['remaining_iteration'] = float(job_dict['iterations'])
            if 'iteration_time' not in job_dict:
                job_dict['iteration_time'] = float(job_dict['duration'])/float(job_dict['iterations'])
            job_dict['iteration_time_cur'] = copy.deepcopy(job_dict['iteration_time'])
            if len(job_dict['resource_time']) == 0:
                tmp_resource = [random.uniform(1.0, 10.0) for i in range(FLAGS.multi_resource)]
                tmp_resource_sum = sum(tmp_resource)
                tmp_resource_time = [tmp_resource[i]/tmp_resource_sum * float(job_dict['iteration_time']) for i in range(FLAGS.multi_resource) ]
                tmp_resource_time[-1] = float(job_dict['iteration_time']) - sum(tmp_resource_time[:-1])
                job_dict['resource_time'] = copy.deepcopy(tmp_resource_time)
            else:
                for i in range(len(job_dict['resource_time'])):
                    job_dict['resource_time'][i] /= 1000

        if 'submit_time' in job_dict:
            job_dict['r_submit_time'] = int(-1 * job_dict['submit_time'])
        if 'antman' in FLAGS.schedule:
            if 'priority' not in job_dict:
                job_dict['priority'] = random.randint(0,1)
            if 'cpu_util' not in job_dict:
                if job_dict['priority']==0:
                    job_dict['cpu_util'] = 0.1 # not real
                else:
                    job_dict['cpu_util'] = 0.9

        job_dict['start_time'] = sys.maxsize
        job_dict['end_time'] = 0
        job_dict['pending_time'] = 0

        job_dict['packing_used'] = 0 # 0 - not used; 1 - prepare for packing; 2 - used

        job_dict['execution_time'] = 0
        job_dict['last_start_time'] = 0
        job_dict['last_check_time'] = 0
        job_dict['executed_time'] = 0
        job_dict['remaining_iterations'] = job_dict['iterations']

        job_dict['preempt'] = 0
        job_dict['resume'] = 0
        job_dict['promote'] = 0
        job_dict['job_counter'] = 0
        job_dict['packing'] = None

        job_dict['status'] = 'ADDED'
        job_dict['job_idx'] = len(self.job_list)
        

        job_dict['cpus'] = list()
        job_dict['placements'] = list() #prepare an empty job_placement 
        job_dict['ps_placements'] = list()
        job_dict['w_placements'] = list()
        job_dict['remaining_cpu'] = job_dict['num_cpu']
        job_dict['last_node_id'] = None
        
        
        if 'model_scale' not in job_dict:
            job_dict['model_scale'] = 1
        #get detailed model inforamtion
        self.get_job_model(job_dict)

        #add job ps/worker information
        self.get_network_load(job_dict)

        self.job_list.append(job_dict)
        self.num_job += 1

        if FLAGS.schedule == 'multi-dlas-cpu':
            num_cpu = job_dict['num_cpu']
            if num_cpu not in self.cpu_job:
                # add that job class
                self.cpu_job[num_cpu] = self.g_job(num_cpu)

            self.cpu_job[num_cpu].total_job += 1
        
    def find_runnable_job(self, job_idx):
        for job in self.runnable_jobs:
            if job['job_idx'] == job_idx:
                return job
        print(f'Not found {job_idx} in runnable_jobs.')
        print([job['job_idx'] for job in self.runnable_jobs])
        assert 1==0

    def read_job_info(self, job_idx, field=None):
       
        print('  Job[%d]: ' % job_idx)

        for job in self.job_list:
            if job['job_idx'] == job_idx:
                #find the job
                if field:
                    if isinstance(job[field], int):
                        print('%s :  %d' % (field, job[field]))
                    else:
                        print('%s :  %s' % (field, job[field]))
                else:
                    print(job)
                print('')

    def read_all_jobs(self, field=None):
        for j in self.job_list:
            print('  Job[%d]: ' % j['job_idx'])
            if field:
                if isinstance(j[field], int):
                    print('%s :  %d' % (field, j[field]))
                else:
                    print('%s :  %s' % (field, j[field]))
            else:
                print(j)
            print('')

    def sort_all_jobs(self, mode=None):

        self.job_list.sort(key = lambda e:e.__getitem__('submit_time'))
        utils.print_fn('   Jobs are sorted with their start time')
        # self.read_all_jobs()
        if FLAGS.schedule == 'multi-dlas-cpu' and FLAGS.scheme == 'count':
            for num_cpu, gjob in self.cpu_job.items():
                utils.print_fn('%d-cpu jobs have %d ' % (num_cpu, gjob.total_job))

    def create_multi_nodes_placement(self, job, switch_id, node_list):
        tmp_dict = dict() 
        tmp_dict['switch'] = switch_id
        tmp_dict['nodes'] = node_list
        job['placements'].append(tmp_dict)

    def create_multi_nodes_placement_same_switch(self, job, switch_id, node_list):
        if len(job['placements'])==0:
            self.create_multi_nodes_placement(job, switch_id, node_list)       
        else:
            for placement in job['placements']:
                if placement['switch'] == switch_id:
                    placement['nodes'].extend(node_list)


    def create_single_node_placement(self, job, switch_id, node_id, num_cpu, mem=0, cpu_list=[], not_first=False):
       
        if not_first:
            node_dict = job['placements'][0]['nodes'][0]
            node_dict['num_cpu']+=num_cpu
            node_dict['mem']+=mem
            # print(job['job_idx'], job['placements'][0]['nodes'][0])
        else:
            tmp_dict = dict() 
            tmp_dict['switch'] = switch_id
            node_dict = dict()
            node_dict['id'] = node_id
            node_dict['num_cpu'] = num_cpu
            node_dict['mem'] = mem
            node_dict['tasks'] = list()
            # node_dict['network'] = round(sum(job['w_network']) + sum(job['ps_network']), 1)
            node_dict['network'] = 0 #single machine, no network traffic

            tmp_dict['nodes'] = list()
            tmp_dict['nodes'].append(node_dict)
            job['placements'].append(tmp_dict)

        return node_dict['network']

    def remove_from_pending(self, job, event_time):
        job['status'] = 'RUNNING'
        job['start_time'] = event_time
        job['end_time'] = job['start_time'] + job['duration']
        job['pending_time'] = job['start_time'] - job['submit_time']

        self.pending_jobs.remove(job)

    def move_to_pending(self, job):
        job['status'] = 'PENDING'
        self.pending_jobs.append(job)


    def update_pending_time(self, event_time):
        for job in self.pending_jobs:
            if 'sumbit_time' in job:
                job['pending_time'] = int(event_time - job['submit_time'])

    def add_to_runnable(self, job):
        job['status'] = 'PENDING'
        self.runnable_jobs.append(job)

    def push_job_to_running(self, job, event_time):
        if job['status'] != 'PENDING':
            return
        job['status'] = 'RUNNING'
        if job['start_time'] == 0:
            job['start_time'] = event_time
        job['last_start_time'] = event_time


    def sort_shortest_runnable_jobs(self, event_time):
        for job in self.runnable_jobs:
            if job['status'] == 'RUNNING':
                new_execution_time = int(event_time - job['last_check_time'])
                job['execution_time'] = int(job['execution_time'] + new_execution_time)
                job['remaining_time'] = int(job['duration'] - job['execution_time'])

            elif job['status'] == 'PENDING':
                job['execution_time'] = 0
                job['remaining_time'] = int(job['duration'])

            job['last_check_time'] = int(event_time)

        JOBS.runnable_jobs.sort(key = lambda e:e.__getitem__('remaining_time'))

    def move_to_runnable(self, job):
        ''' job gets into the system: pending or running, and finally END'''
        #job not started yet
        job['status'] = 'PENDING'
        job['start_time'] = sys.maxsize
        job['last_start_time'] = 0
        job['last_check_time'] = job['submit_time']
        job['total_executed_time'] = 0 # total
        job['total_executed_cputime'] = 0
        job['calc_executed_time'] = 0
        job['executed_time'] = 0 # used for deciding priority queue, may be zeroed by last_pending_time
        job['pending_time'] = 0
        job['last_pending_time'] = 0 # how much pending_time the job has since last entering the highest priority queue

        if FLAGS.schedule == 'multi-dlas-cpu':
            num_cpu = job['num_cpu']
            self.cpu_job[num_cpu].runnable_jobs.append(job)
        elif 'multi-resource' in FLAGS.schedule or 'antman' in FLAGS.schedule:
            # job['executed_iteration'] = 0
            self.runnable_jobs.append(job)
        else:
            self.runnable_jobs.append(job)
    
    def update_priority_queues(self, cputime=False):
        for queue in self.queues:
            del queue[:]
        for job in self.runnable_jobs:
            if cputime:
                j_gt = int(job['executed_time'] * job['num_cpu'])
            else:
                j_gt = int(job['executed_time'])

            if j_gt < self.queue_limit[0]:
                self.queues[0].append(job)
                job['q_id'] = 0
            else:
                self.queues[1].append(job)
                job['q_id'] = 1

            # elif j_gt < self.queue_limit[1]:
            #     self.queues[1].append(job)
            #     job['q_id'] = 1
            # elif j_gt < self.queue_limit[2]:
            #     self.queues[2].append(job)
            #     job['q_id'] = 2
            # else:
            #     self.queues[3].append(job)
            #     job['q_id'] = 3

   
    def print_job_events(self):
        utils.print_fn('    Print all job events ')
        for event in self.job_events:
            utils.print_fn('      event.time[%d], with %d start_jobs, and %d end_jobs' % 
                            (event['time'], len(event['start_jobs']), len(event['end_jobs'])))

        utils.print_fn(' ')

    def add_job_end_event(self, job):
        #for job end 
        tmp_dict = utils.search_dict_list(self.job_events, 'time', job['end_time'])
        if tmp_dict == None:
            #not found, add the time into to job_events
            tmp_dict = dict()
            tmp_dict['time'] = job['end_time']
            tmp_dict['start_jobs'] = list()
            tmp_dict['end_jobs'] = list()
            tmp_dict['end_jobs'].append(job)
            self.job_events.append(tmp_dict)
        else:
            tmp_dict['end_jobs'].append(job)



    def prepare_job_start_events(self):
        for job in self.job_list:
            start_t = job['submit_time']
            # utils.print_fn('%d, %d' % (start_t, end_t))

            #for job start
            tmp_dict = utils.search_dict_list(self.job_events, 'time', start_t)
            if tmp_dict == None:
                #not found, add the time into to job_events
                tmp_dict = dict()
                tmp_dict['time'] = start_t
                tmp_dict['start_jobs'] = list()
                tmp_dict['end_jobs'] = list()
                tmp_dict['start_jobs'].append(job)
                self.job_events.append(tmp_dict)
            else:
                tmp_dict['start_jobs'].append(job)


            job['status'] = 'EVENT' #job has been in EVENT status

        ''' sort events based on their time'''
        self.job_events.sort(key = lambda e:e.__getitem__('time'))
        utils.print_fn('Init, add job start events')
        self.print_job_events()


    def add_migratable(self, job):
       
        if job['num_w'] <= 1:
            return

        if job not in self.migratable_jobs:
            self.migratable_jobs.append(job)            


    def remove_migratable(self, job):
        '''
        remove from migratable job list

        '''
        if job in self.migratable_jobs:
            self.migratable_jobs.remove(job)

    def print_placement(self, ejob):
        print("placement of job ", ejob['job_idx'])
        print(ejob['placements'])

    def to_jobinfo(self, tmp_ejob, is_packing=False):
        jobinfo = None

        cpu_list = {}
        if not is_packing:
            ejob = [tmp_ejob]
            while len(ejob)<FLAGS.multi_resource:
                ejob.append(None)
        else:
            ejob = tmp_ejob
        # if len(ejob[0]['placements'])!=1:
        #     print(ejob[0])
        assert len(ejob[0]['placements'])==1
        placement = ejob[0]['placements'][0]
        job_id_list = [rjob['job_idx'] if rjob!=None else -1 for rjob in ejob]
        job_name_list = [rjob['model_name'] if rjob!=None else '0' for rjob in ejob]
        batch_size_list = [rjob['batch_size'] if rjob!=None else 0 for rjob in ejob]
        if FLAGS.fast_forwarding>0:
            iters_list0 = [rjob['remaining_iterations'] if rjob!=None else 0 for rjob in ejob]
            iters_sorted = sorted(list(set(iters_list0)))
            tmp_iter = 5
            # print(iters_list0, iters_sorted)
            iters_list = [0,0,0,0]
            num_jobs = sum([1 if rjob!=None else 0 for rjob in ejob])
            for iters in iters_sorted:
                if iters==0:
                    continue
                tmp_iter += int(FLAGS.fast_forwarding/num_jobs)
                for idx, iter0 in enumerate(iters_list0):
                    if iter0 == iters:
                        iters_list[idx] = tmp_iter
            last_iters = list(set(iters_list))
            last_iters.sort()
            # print('jobs, to_info:', last_iters)
            if last_iters[0]==0:
                del last_iters[0]
            for rjob in ejob:
                if rjob!=None:
                    rjob['last_iters'] = last_iters
        else:
            iters_list = [rjob['remaining_iterations'] if rjob!=None else 0 for rjob in ejob]
        job_counter_list = [rjob['job_counter'] if rjob!=None else 0 for rjob in ejob]
        job_num = len(ejob)
        node_id_list = []
        for node in placement['nodes']:
            node_id = node['id']
            node_id_list.append(node_id)
            assert node_id not in cpu_list
            assert len(placement['nodes'])==1 or (len(placement['nodes'])>1 and len(node['cpu_list'])==8)
        jobinfo = JobInfo(num=job_num, cpus=cpu_list[node_id_list[0]], num_cpu=ejob[0]['num_cpu'])
        jobinfo.node_id.extend(node_id_list)
        jobinfo.job_id.extend(job_id_list)
        jobinfo.job_name.extend(job_name_list)
        jobinfo.batch_size.extend(batch_size_list)
        jobinfo.iterations.extend(iters_list)
        jobinfo.job_counter.extend(job_counter_list)
        
        return jobinfo

    def calc_packing_finished_info(self, rjob, tmp_time, last_check_time):
        iter_list = list()
        # print('in calc_packing_info: ', rjob['job_idx'], [tjob.job_idx for tjob in rjob['packing'].packing_jobs])
        if rjob['packing']==None:
            iter_list.append(rjob['remaining_iterations'])
        else:
            for pjob_mini in rjob['packing'].packing_jobs:
                pjob=self.find_runnable_job(pjob_mini.job_idx)
                iter_list.append(pjob['remaining_iterations'])
            sim_itertime = rjob['packing'].calc_iteration_time()
            real_itertime = rjob['real_itertime'][0]
            overhead_error = (real_itertime-sim_itertime)/sim_itertime
            self.overhead_list[len(rjob['packing'].packing_jobs)].append(overhead_error)
                # print(pjob['job_idx'], pjob['real_itertime'], pjob['remaining_iterations'])
        iter_list = list(set(iter_list))
        iter_list.sort()
        if iter_list[0]==0:
            del iter_list[0]
        # print('calc_packing_finished_info, real_itertime vs iter_list: ', iter_list)
        assert len(rjob['real_itertime']) == len(iter_list)
        finished_iteration = 0
        if rjob['last_finish_time']>last_check_time:
            overhead_time = copy.deepcopy(rjob['last_finish_time'])
            # print('calc jobs: ', rjob['job_idx'], rjob['last_iters'])
            last_iter = 0
            assert len(rjob['last_iters']) == len(rjob['real_itertime'])
            for idx, itertime in enumerate(rjob['real_itertime']):
                overhead_time -= (rjob['last_iters'][idx]-last_iter) * itertime
                last_iter = rjob['last_iters'][idx]
        else:
            overhead_time = last_check_time
        remaining_time = tmp_time - overhead_time
        finished_time = overhead_time
        done_idx = 0
        last_iter = 0
        # print('calc jobs: ', rjob['job_idx'], overhead_time-last_check_time)
        for idx, iters in enumerate(iter_list):
            if iters > rjob['remaining_iterations']:
                break
            time0 = (iters-last_iter)*rjob['real_itertime'][idx]
            if remaining_time - time0>=0:
                remaining_time -= time0
                finished_time += time0
                finished_iteration += (iters-last_iter)
                done_idx += 1
                last_iter = iters
            else:
                finished_time = tmp_time
                finished_iteration += int(remaining_time / rjob['real_itertime'][idx])
                break
        
        assert finished_iteration<=rjob['remaining_iterations']
        return finished_time, finished_iteration, done_idx


JOBS = _TFJobs()

_allowed_symbols = [
    'JOBS'
]
