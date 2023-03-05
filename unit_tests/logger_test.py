import os 
os.sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
import numpy as np
from saferl_plotter import log_utils as lu

class RunTimeLogger():
    """
    Logger for runtime information.
    Note that this class is not aiming for saving logging information or experiment-name related information.
    """
    def __init__(self):
        self.logger = {}
    
    @property
    def len(self):
        log_len = len(self.logger[list(self.logger.keys())[0]])
        return log_len
    
    @property
    def empty(self):
        log_len = len(self.logger[list(self.logger.keys())[0]])
        if log_len == 0:
            return True
        else:
            return False
        
    def update(self, **kwargs):
        """
        Update the logger.
        """
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
            
'''
test the update function
'''
# logger = RunTimeLogger()
# state = np.random.rand(2,3)
# action = np.random.rand(1,3)
# next_state = np.random.rand(2,3)
# info = {"state": state, "action": action, "next_state": next_state}
# logger.update(**{"state": state, "action": action, "next_state": next_state})
# logger.update(**{"state": state, "action": action, "next_state": next_state})
# logger.update(**{"state": state, "action": action, "next_state": next_state})
# logger.update(**{"state": state, "action": action, "next_state": next_state})
# logger.update(**{"state": state, "action": action, "next_state": next_state})
# import ipdb; ipdb.set_trace()

'''
test the update value to go function
'''
logger = RunTimeLogger()
reward = np.array([i for i in range(10)])
cost = np.array([0.1*i for i in range(10)])
for i in range(10):
    logger.update(**{"reward": reward[i], "cost": cost[i]}) 
logger.update_value_to_go("reward", discount=0.9)
logger.update_value_to_go("cost", discount=0.9)
import ipdb; ipdb.set_trace()
print(logger.get_complete_stats())
