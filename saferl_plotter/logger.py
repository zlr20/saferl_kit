#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Linrui Zhang'

import csv
import os
import json, time
import numpy as np
from saferl_plotter import log_utils as lu

class RunTimeLogger():
    """
    Logger for runtime information.
    Note that this class is not aiming for saving logging information or experiment-name related information.
    """
    def __init__(self):
        self.logger = {}
        self.empty = True
    
    @property
    def len(self):
        log_len = len(self.logger[list(self.logger.keys())[0]])
        return log_len
        
    def update(self, **kwargs):
        # update the logger status
        if self.empty:
            self.empty = False
            
        for key, value in kwargs.items():
            if self.logger.get(key) is None:
                # update the logger 
                self.logger.update({key: []})
                self.logger[key].append(value)
            else:
                self.logger[key].append(value)
        
        # make sure all fields have the same length
        log_len = len(self.logger[list(self.logger.keys())[0]])
        for key in self.logger.keys():
            try:
                assert len(self.logger[key]) == log_len
            except AssertionError:
                print(lu.colorize(f"Not all keys are upated, please make sure the logging information is consistent for each update", 'yellow', bold=True))
            
    def get_stats(self, key):
        """
        Return the mean and std of the given key.
        """
        v = self.logger[key]
        vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape)>0 else v
        mean = np.mean(vals)
        std = np.std(vals)
        
        return mean, std
    
    def get_complete_stats(self):
        """
        Return the complete stats for all keys.
        """
        return self.logger
    
    def get(self, key):
        """
        get logger for the given key.
        """
        return self.logger[key]
    
    def reset(self):
        """
        Refresh the logger. Clear all the data. Called after each episode.
        """
        self.logger = {}
        
        
    def update_value_to_go(self, key, discount=1):
        """
        update the value to go for a given key.
        """
        value_trajectory = self.logger[key]
        value_to_go = [0.] * len(value_trajectory)
        for i in range(len(value_trajectory)):
            value_to_go[i] = value_trajectory[i]
            for j in range(i+1, len(value_trajectory)):
                value_to_go[i] += discount**(j-i) * value_trajectory[j]
        new_key = f"{key}_to_go"
        self.logger.update({new_key: value_to_go})
        
        # make sure all fields have the same length
        log_len = len(self.logger[list(self.logger.keys())[0]])
        for key in self.logger.keys():
            try:
                assert len(self.logger[key]) == log_len
            except AssertionError:
                print(lu.colorize(f"Not all keys are upated, please make sure the logging information is consistent for each update", 'yellow', bold=True))
            
        
        
        


class Logger():
    def __init__(self, log_dir="./logs", exp_name=None, env_name=None, seed=0, config=None, filename='evaluator.csv', debug=False):
        self.exp_name = exp_name
        self.env_name = env_name
        self.seed = seed
        self.previous_log_time = time.time()
        self.start_log_time = self.previous_log_time
        self.debug = debug

        if debug:
            print(lu.colorize(f"\nDebug mode is activate !!!\nLog will NOT be saved !!!\n", 'red', bold=True))
        else:
            num_exps = 0
            self.log_dir = f"./{log_dir}/{exp_name.replace('-', '_')}_{env_name.replace('-', '_')}-seed{seed}"
            while True:
                if os.path.exists(f"{self.log_dir}-{str(num_exps)}/"):
                    num_exps += 1
                else:
                    self.log_dir += f"-{str(num_exps)}/"
                    os.makedirs(self.log_dir)
                    break
            self.csv_file = open(self.log_dir + '/' + filename, 'w', encoding='utf8')
            # header={"t_start": time.time(), 'env_id' : env_name, 'exp_name': exp_name, 'seed': seed}
            # header = '# {} \n'.format(json.dumps(header))
            # self.csv_file.write(header)
            self.logger = csv.DictWriter(self.csv_file, fieldnames=('mean_score', 'total_steps', 'std_score', 'max_score', 'min_score'))
            self.logger.writeheader()
            self.csv_file.flush()

            if config != None:
                lu.save_config(exp_name, config, self.log_dir)


    def update(self, score, total_steps):
        '''
            Score is a list
        '''
        current_log_time = time.time()
        avg_score = np.mean(score)
        std_score = np.std(score)
        max_score = np.max(score)
        min_score = np.min(score)

        print(lu.colorize(f"\nTime: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}, Time spent from previous logger: {(current_log_time - self.previous_log_time):.3f} s", 'yellow', bold=True))
        print(lu.colorize(f"Evaluation over {len(score)} episodes after {total_steps}:", 'yellow', bold=True))
        print(lu.colorize(f"Avg: {avg_score:.3f} Std: {std_score:.3f} Max: {max_score:.3f} Min: {min_score:.3f}\n", 'yellow', bold=True))
        self.previous_log_time = current_log_time
        
        if not self.debug:
            epinfo = {"mean_score": avg_score, "total_steps": total_steps, "std_score": std_score, "max_score": max_score, "min_score": min_score}
            self.logger.writerow(epinfo)
            self.csv_file.flush()
    

class SafeLogger():
    def __init__(self, log_dir="./logs", exp_name=None, env_name=None, seed=0,config=None, filename="logger.csv", fieldnames=[], debug=False):
        self.exp_name = exp_name
        self.env_name = env_name
        self.seed = seed
        self.previous_log_time = time.time()
        self.start_log_time = self.previous_log_time
        self.debug = debug
        self.fieldnames = ["total_steps"] + fieldnames

        if debug:
            print(lu.colorize(f"\nDebug mode is activate !!!\nLog will NOT be saved !!!\n", 'red', bold=True))
        else:
            num_exps = 0
            self.log_dir = f"./{log_dir}/{exp_name.replace('-', '_')}_{env_name.replace('-', '_')}-seed{seed}"
            while True:
                if os.path.exists(f"{self.log_dir}-{str(num_exps)}/"):
                    num_exps += 1
                else:
                    self.log_dir += f"-{str(num_exps)}/"
                    os.makedirs(self.log_dir)
                    break     
            self.csv_file = open(self.log_dir + '/' + filename, 'w', encoding='utf8')
            # header={"t_start": time.time(), 'env_id' : env_name, 'exp_name': exp_name, 'seed': seed}
            # header = '# {} \n'.format(json.dumps(header))
            # self.csv_file.write(header)
            self.logger = csv.DictWriter(self.csv_file, fieldnames=(self.fieldnames))
            self.logger.writeheader()
            self.csv_file.flush()

            if config != None:
                lu.save_config(exp_name, config, self.log_dir)


    def update(self, fieldvalues, total_steps):
        epinfo = {}
        fieldvalues = [total_steps] + fieldvalues

        current_log_time = time.time()
        print(lu.colorize(f"\nTime: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}, Time spent from previous logger: {(current_log_time - self.previous_log_time):.3f} s", 'blue', bold=True))
        print(lu.colorize(f"CustomLogger with fileds: {self.fieldnames}", 'blue', bold=True))
        print(lu.colorize(f"fieldvalues: {fieldvalues}\n", 'blue', bold=True))
        self.previous_log_time = current_log_time
       
        if not self.debug:
            for filedname, filedvalue in zip(self.fieldnames, fieldvalues):
                epinfo.update({filedname: filedvalue})
            self.logger.writerow(epinfo)
            self.csv_file.flush()
    