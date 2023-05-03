import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np

DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()
# color_map = {'scpo': 'tab:green', 'trpolag': 'tab:blue', 'C': 'green'}
# algos = ["cpo","pcpo","safelayer","scpo","trpo","trpofac","trpoipo","trpolag","usl"]
# algos = ["cpo","pcpo","safelayer","scpo","trpo","trpofac","trpoipo","trpolag"]
# algos = ["trpo", "trpolag", "safelayer", "usl", "trpoipo", "trpofac", "cpo", "pcpo", "scpo"]
algos = ["scpo"]
cmap = plt.get_cmap('tab20')
cmap = ['tab:blue', 
        'tab:orange', 
        'tab:green', 
        'tab:purple', 
        'tab:brown', 
        'tab:pink', 
        'tab:gray', 
        'tab:olive', 
        'tab:cyan',
        'tab:red']
cmap = plt.get_cmap('tab20')(np.linspace(0, 1, 20))
table_name = ["TRPO", "TRPO-Lagrangian", "TRPO-SL", "TRPO-USL", "TRPO-IPO", "TRPO-FAC", "CPO", "PCPO", "SCPO"]
tasks = []
# tasks = ['goal1','goal4','goal8',
#              'goal1_pillar','goal4_pillar','goal8_pillar', 
#              'swimmertiny1','swimmertiny4','swimmertiny8',
#              'drone1', 'drone4', 'drone8',]
tasks = tasks + ['walker8','anttiny8']
# Generate a list of 20 distinct colors
color_map = {}
table_map = {}
for i in range(len(algos)):
    table_map[tasks[i]] = tasks[i]
    color_map[table_name[i]] = cmap[i]
def plot_data(data, xaxis='Epoch', value="AverageEpRet", condition="Condition1", smooth=1, **kwargs):
    # if value == 'Cost_Performance':
    #     smooth = 50
    # if smooth > 1:
    #     """
    #     smooth data with moving window average.
    #     that is,
    #         smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
    #     where the "smooth" param is width of that window (2k+1)
    #     """
    #     y = np.ones(smooth)
    #     for datum in data:
    #         x = np.asarray(datum[value])
    #         z = np.ones(len(x))
    #         smoothed_x = np.convolve(x,y,'same') / np.convolve(z,y,'same')
    #         datum[value] = smoothed_x
    y_max = -1
    
    # for datum in data:
    #     x = np.asarray(datum[xaxis])
    #     y = np.asarray(datum[value])
    #     max_x = max(x)
    #     max_y = max(y)
    #     # z = np.ones(len(x))
    #     # smoothed_x = np.convolve(x,y,'same') / np.convolve(z,y,'same')
    #     datum[xaxis] = x / max_x
    #     datum[value] = y / max_y
    # smooth = 1
    # y = np.ones(smooth)
    # for datum in data:
    #     x = np.asarray(datum[value])
    #     z = np.ones(len(x))
    #     smoothed_x = np.convolve(x,y,'same') / np.convolve(z,y,'same')
    #     datum[value] = smoothed_x
    
    # for datum in data:
    #     if datum['Condition3'][0] == "TRPO-SL":
    #         continue
    #     y_max = max(y_max, max(datum[value]))
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    y_max = max(data[value])
    y_min = min(data[value])
    print(set(data["Condition3"]))
    condition="Condition3"
    sns.set(style="darkgrid", font_scale=1.5)
    # print(value)
    # print(data[value])
    # sns.tsplot(data=data, time=xaxis, value=value, color = color_map, unit="Unit", condition=condition, ci='sd', **kwargs)
    sns.tsplot(data=data, time=xaxis, value=value, unit="Unit", condition=condition, ci='sd', **kwargs)
    """
    If you upgrade to any version of Seaborn greater than 0.8.1, switch from 
    tsplot to lineplot replacing L29 with:

        sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci='sd', **kwargs)

    Changes the colorscheme and the default legend style, though.
    """
    

    """
    For the version of the legend used in the Spinning Up benchmarking page, 
    swap L38 with:

    plt.legend(loc='upper center', ncol=6, handlelength=1,
               mode="expand", borderaxespad=0., prop={'size': 13})
    """
    
    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    # plt.ylim([0,1])
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    if value == 'Reward_Performance':
        plt.legend(loc='lower right',ncol=3, handlelength=2,mode="expand",borderaxespad=0., prop={'size': 13}).set_draggable(True)
    else:
        plt.legend(loc='upper right',ncol=1, handlelength=2, borderaxespad=0., prop={'size': 13}).set_draggable(True)
    #plt.legend(loc='upper center', ncol=3, handlelength=1,
    #           borderaxespad=0., prop={'size': 13})
    # plt.xlabel("Percentage of total epochs")
    plt.ylim([y_min-0.1*(y_max - y_min), y_max*1.1])
    plt.ylabel("Maximum State-wise Cost")
    plt.tight_layout(pad=0.5)

def get_datasets(logdir, condition=None):
    """
    Recursively look through logdir for output files produced by
    spinup.logx.Logger. 

    Assumes that any file "progress.txt" is a valid hit. 
    """
    global exp_idx
    global units
    datasets = []
    for root, _, files in os.walk(logdir):

        if 'progress.txt' in files:
            exp_name = None
            try:
                config_path = open(os.path.join(root,'config.json'))
                config = json.load(config_path)
                if 'exp_name' in config:
                    exp_name = config['exp_name']
            except:
                print('No file named config.json')
            condition1 = condition or exp_name or 'exp'
            condition2 = condition1 + '-' + str(exp_idx)
            for x in tasks:
                algo = x + '_'
                print(algo, condition1)
                if algo in condition1:
                    if 'walker8' in x:
                        condition3 = 'Walker'
                    elif 'ant' in x:
                        condition3 = 'Ant'
                    else:
                        condition3 = x
                   
            
            exp_idx += 1
            if condition1 not in units:
                units[condition1] = 0
            unit = units[condition1]
            units[condition1] += 1

            try:
                exp_data = pd.read_table(os.path.join(root,'progress.txt'))
            except:
                print('Could not read from %s'%os.path.join(root,'progress.txt'))
                continue
            reward_performance = 'EpRet' if 'EpRet' in exp_data else 'AverageEpRet'
            cost_performance = 'EpCost' if 'EpCost' in exp_data else 'AverageEpCost'
            max_cost_performance = 'EpMaxCost' if 'EpMaxCost' in exp_data else 'None'
            print(max_cost_performance)
            cost_rate_performance = 'AverageTestCostRate' if 'AverageTestCostRate' in exp_data else 'CostRate'
            exp_data.insert(len(exp_data.columns),'Unit',unit)
            exp_data.insert(len(exp_data.columns),'Condition1',condition1)
            exp_data.insert(len(exp_data.columns),'Condition2',condition2)
            try:
                exp_data.insert(len(exp_data.columns),'Condition3',condition3)
            except:
                continue
            exp_data.insert(len(exp_data.columns),'Reward_Performance',exp_data[reward_performance])
            # if exp_data[cost_performance]:
            if cost_performance in exp_data:
                exp_data.insert(len(exp_data.columns),'Cost_Performance',exp_data[cost_performance])
            if cost_rate_performance in exp_data:
                exp_data.insert(len(exp_data.columns),'Cost_Rate_Performance',exp_data[cost_rate_performance])
            if max_cost_performance in exp_data:
                exp_data.insert(len(exp_data.columns),'Max_Cost_Performance',exp_data[max_cost_performance])
            datasets.append(exp_data)   
    return datasets

def selectLogs(task):
    logdirs = []
    select = [task + '_noconti']
    # if task in ['walker8', 'anttiny8']:
    #     select.append('1000')
    algos = ['scpo']
    for algo in algos:
        print(algo)
        algodir = './' + algo + '/logs'
        if algo == 'scpo':
            logs = ['drone1_noconti_scpo_adascale_scale0.03_step30000_epochs200',
                    'drone4_noconti_scpo_adascale_scale0.1_step30000_epochs200',
                    'drone8_noconti_scpo_adascale_kl0.02_scale0.03',
                    'goal1_noconti_scpo_adascale_scale0.1_step30000',
                    'goal4_noconti_scpo_adascale_scale0.03_step30000',
                    'goal8_noconti_scpo_adascale_kl0.02_scale0.1',
                    'goal1_pillar_noconti_scpo_fixed_kl0.02_target_cost-0.3_epoch200',
                    'goal4_pillar_noconti_scpo_fixed_kl0.02_target_cost-0.5_epoch200',
                    'goal8_pillar_noconti_scpo_fixed_kl0.02_target_cost-0.5',
                    'swimmertiny1_noconti_scpo_adascale_scale0.1_step30000_epochs200',
                    'swimmertiny4_noconti_scpo_adascale_scale0.1_step30000_epochs200',
                    'swimmertiny8_noconti_scpo_adapt_kl0.02_scale0.001',
                    'walker8_noconti_scpo_fixed_kl0.02_target_cost-0.1_epoch1000',
                    'anttiny8_noconti_scpo_fixed_kl0.02_target_cost-0.1_epoch1000',
                    ]
        else:
            logs = os.listdir(algodir)
            
        if algo == 'safelayer':
            logs = [log for log in logs if ('downsample' in log and 'nowarm' not in log)]
        if algo == 'usl':
            logs = [log for log in logs if ('mc' not in log)]
        # logs = [log for log in logs if ('debug' not in log)]
            
        logdirs_target = [log for log in logs if all(x in log for x in select)]
        logdirs_target = logs
        
        print(logdirs_target)
        for i in range(len(logdirs_target)):
            log_target = algodir + '/' + logdirs_target[i] + '/'
            logdirs.append(log_target)
    return logdirs

def get_all_datasets(all_logdirs, legend=None, select=None, exclude=None, task=None):
    """
    For every entry in all_logdirs,
        1) check if the entry is a real directory and if it is, 
           pull data from it; 

        2) if not, check to see if the entry is a prefix for a 
           real directory, and pull data from that.
    """
    logdirs = []
    
    if task is not None:
        print(task)
        logdirs = selectLogs(task)
    else:
        for logdir in all_logdirs:
            if osp.isdir(logdir) and logdir[-1]==os.sep:
                logdirs += [logdir]
            else:
                basedir = osp.dirname(logdir)
                fulldir = lambda x : osp.join(basedir, x)
                prefix = logdir.split(os.sep)[-1]
                listdir= os.listdir(basedir)
                logdirs += sorted([fulldir(x) for x in listdir if prefix in x])

        """
        Enforce selection rules, which check logdirs for certain substrings.
        Makes it easier to look at graphs from particular ablations, if you
        launch many jobs at once with similar names.
        """
        if select is not None:
            logdirs = [log for log in logdirs if all(x in log for x in select)]
        if exclude is not None:
            logdirs = [log for log in logdirs if all(not(x in log) for x in exclude)]

    

    # Verify logdirs
    print('Plotting from...\n' + '='*DIV_LINE_WIDTH + '\n')
    for logdir in logdirs:
        print(logdir)
    print('\n' + '='*DIV_LINE_WIDTH)

    # Make sure the legend is compatible with the logdirs
    assert not(legend) or (len(legend) == len(logdirs)), \
        "Must give a legend title for each set of experiments."

    # Load data from logdirs
    data = []
    if legend:
        for log, leg in zip(logdirs, legend):
            data += get_datasets(log, leg)
    else:
        for log in logdirs:
            print(log)
            data += get_datasets(log)
    return data


def make_plots(all_logdirs, legend=None, xaxis=None, values=None, count=False,  
               font_scale=1.5, smooth=1, select=None, exclude=None, estimator='mean',
               results_dir=None, title='reward', reward_flag=True, cost_flag=False,
               task=None):
    print(task)
    # create a separate folder for each plot 
    # results_dir = osp.join(results_dir, title)
    data = get_all_datasets(all_logdirs, legend, select,exclude, task)
    # result = {}
    # for d in data:
    #     last_d = d.iloc[-1]
    #     algo_name = last_d['Condition3']
    #     algo_reward = last_d['Reward_Performance']
    #     algo_cost = last_d['Cost_Performance']
    #     algo_cost_rate = last_d['Cost_Rate_Performance']
    #     if algo_name not in result:
    #         result[algo_name] = {}
    #         result[algo_name]['reward'] = []
    #         result[algo_name]['cost'] = []
    #         result[algo_name]['cost_rate'] = []
    #     result[algo_name]['reward'].append(algo_reward)
    #     result[algo_name]['cost'].append(algo_cost)
    #     result[algo_name]['cost_rate'].append(algo_cost_rate)
    
    # final_result = []
    # algo_name_list = []
    
    
    # for algo_name in result:
    #     r = np.around(np.mean(result[algo_name]['reward']),8)
    #     c = np.around(np.mean(result[algo_name]['cost']),8)
    #     cr = np.around(np.mean(result[algo_name]['cost_rate']),8)
    #     final_result.append([r,c,cr])
    #     algo_name_list.append(algo_name)
    #     output = algo_name + ' & ' + str(r) + ' & ' + str(c) + ' & ' + str(cr) + '\\\\' + '\n'
        
    #     print(algo_name, r, c, cr)
    # max_idx = np.argmax(final_result, axis=0)
    # min_idx = np.argmin(final_result, axis=0)
    # file_name = "results/scpo_paper_table/"+ task + ".txt"
    # f = open(file_name, "w")
    # for i in range(len(final_result)):
    #     output = algo_name_list[i]
    #     r,c,cr = final_result[i]
    #     np.set_printoptions(precision=3)
    #     if i == max_idx[0]:
    #         output += ' & \\textbf{' + '%.04f' %(r) + '}'
    #     else:
    #         output += ' & ' + '%.04f' %(r)
    #     if i == min_idx[1]:
    #         output += ' & \\textbf{' + '%.04f' %(c) + '}'
    #     else:
    #         output += ' & ' + '%.04f' %(c)
    #     if i == min_idx[2]:
    #         output += ' & \\textbf{' + '%.04f' %(cr) + '}'
    #     else:
    #         output += ' & ' + '%.04f' %(cr)
    #     output += '\\\\' + '\n'
        
    #     print(output)
    #     f.write(output)
    # f.close()
    # print(result)
    # return
    # values = values if isinstance(values, list) else [values]
    values = []
    # if reward_flag:
    #     values.append('Reward_Performance')
    # if cost_flag:
    #     values.append('Cost_Performance')
    #     values.append('Cost_Rate_Performance')
    values.append('Max_Cost_Performance')
    
    condition = 'Condition2' if count else 'Condition1'
    estimator = getattr(np, estimator)      # choose what to show on main curve: mean? max? min?
    for value in values:
        subdir = 'max_cost/'
        plt.figure()
        try:
            plot_data(data, xaxis=xaxis, value=value, condition=condition, smooth=smooth, estimator=estimator)
        except:
            print(f"this key {value} is not in the data")
            break
        # make direction for save figure
        final_dir = osp.join(results_dir, subdir)
        existence = os.path.exists(final_dir)
        if not existence:
            os.makedirs(final_dir)
        plt.show()
        plt.savefig(final_dir + value, dpi=400, bbox_inches='tight')
# python utils/plot_yifan_maxC.py comparison_yifan/ --results_dir results/scpo_paper --reward --cost --title pillar1

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--results_dir', default='./results/',
                        help='plot results dir (default: ./)')
    parser.add_argument('--title', default='reward',
                        help='the title for the saved plot')
    parser.add_argument('--legend', '-l', nargs='*')
    parser.add_argument('--xaxis', '-x', default='TotalEnvInteracts')
    parser.add_argument('--value', '-y', default='Performance', nargs='*')
    parser.add_argument('--reward', action='store_true')
    parser.add_argument('--cost', action='store_true')
    parser.add_argument('--count', action='store_true')
    parser.add_argument('--smooth', '-s', type=int, default=1)
    parser.add_argument('--select', nargs='*')
    parser.add_argument('--exclude', nargs='*')
    parser.add_argument('--est', default='mean')
    parser.add_argument('--task', default=None)
    args = parser.parse_args()
    """

    Args: 
        logdir (strings): As many log directories (or prefixes to log 
            directories, which the plotter will autocomplete internally) as 
            you'd like to plot from.

        legend (strings): Optional way to specify legend for the plot. The 
            plotter legend will automatically use the ``exp_name`` from the
            config.json file, unless you tell it otherwise through this flag.
            This only works if you provide a name for each directory that
            will get plotted. (Note: this may not be the same as the number
            of logdir args you provide! Recall that the plotter looks for
            autocompletes of the logdir args: there may be more than one 
            match for a given logdir prefix, and you will need to provide a 
            legend string for each one of those matches---unless you have 
            removed some of them as candidates via selection or exclusion 
            rules (below).)

        xaxis (string): Pick what column from data is used for the x-axis.
             Defaults to ``TotalEnvInteracts``.

        value (strings): Pick what columns from data to graph on the y-axis. 
            Submitting multiple values will produce multiple graphs. Defaults
            to ``Performance``, which is not an actual output of any algorithm.
            Instead, ``Performance`` refers to either ``AverageEpRet``, the 
            correct performance measure for the on-policy algorithms, or
            ``AverageTestEpRet``, the correct performance measure for the 
            off-policy algorithms. The plotter will automatically figure out 
            which of ``AverageEpRet`` or ``AverageTestEpRet`` to report for 
            each separate logdir.

        count: Optional flag. By default, the plotter shows y-values which
            are averaged across all results that share an ``exp_name``, 
            which is typically a set of identical experiments that only vary
            in random seed. But if you'd like to see all of those curves 
            separately, use the ``--count`` flag.

        smooth (int): Smooth data by averaging it over a fixed window. This 
            parameter says how wide the averaging window will be.

        select (strings): Optional selection rule: the plotter will only show
            curves from logdirs that contain all of these substrings.

        exclude (strings): Optional exclusion rule: plotter will only show 
            curves from logdirs that do not contain these substrings.

    """

    reward_flag = True if args.reward else False
    cost_flag = True if args.cost else False
    
    # make_plots(args.logdir, args.legend, args.xaxis, args.value, args.count, 
    #            smooth=args.smooth, select=args.select, exclude=args.exclude,
    #            estimator=args.est, results_dir=args.results_dir, title=args.title,
    #            reward_flag=reward_flag, cost_flag=cost_flag, task=args.task)
    # python utils/plot_yifan.py comparison_yifan/ --results_dir results/scpo_paper --reward --cost --title pillar1 
    tasks = ['walker8']
    # tasks = ['goal1','goal4','goal8',
    #          'goal1_pillar','goal4_pillar','goal8_pillar', 
    #          'swimmertiny1','swimmertiny4','swimmertiny8',
    #          'drone1', 'drone4', 'drone8',]
    # tasks = tasks + ['walker8','anttiny8']
    print(tasks)
    # tasks = ['swimmertiny1','swimmertiny4','swimmertiny8']
    # tasks = ['anttiny8','walker8']
    for task in tasks:
        make_plots(args.logdir, args.legend, args.xaxis, args.value, args.count, 
               smooth=args.smooth, select=args.select, exclude=args.exclude,
               estimator=args.est, results_dir=args.results_dir, title=task,
               reward_flag=reward_flag, cost_flag=cost_flag, task=task)

if __name__ == "__main__":
    main()