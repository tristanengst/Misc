#!/usr/bin/env python3

"""Script to log things from the cluster. Should be run every 10 minutes or so.
It maintains three dicitonaries as JSON files giving the data we care about.

The basic idea is to frequently log the LevelFS of each account, the jobs that
are pending, and the jobs that are running. This allows us to construct a
per-cluster from account LevelFS percentiles when jobs were submitted to the
resources requested for the job (measured in quarters of a node) to the amount
of time requested for the job (measured in quantized hours) to the average time
it took for the job to start. We can use this mapping to figure out what LevelFS
percentile we want to have.

Args:
All arguments are set with intelligent defaults.

--max_unlogged_pending_time  : the maximum time a job could have been pending for us
                            to log it with the last recorded LevelFS for its
                            account in the case where we never see it pending and
                            thus don't have a LevelFS to log for when it was
                            submitted. This should be the same as the interval
                            this script is run at.
--min_level_fs          : the minimum (raw, not percentile) LevelFS we should
                            log data for. We think accounts with very small
                            LevelFS values might get jobs very fast, in
                            contradiction to the stated policies
--level_fs_to_wait      : JSON file to write the mapping described above to
--recent_cluster_data   : JSON file logging recent cluster data
--human_readable        : human-readable script outputs as a .txt file
"""
import argparse
from collections import defaultdict
from datetime import datetime, timedelta
import os
import time
import json

################################################################################
# Global scope contant-y things.
################################################################################
clusters = ["narval"]

time_buckets = [(h, h+3) for h in range(0, 12, 3)] + [[12,18], [18,24]]
time_buckets += [(h, h+12) for h in range(24, 72, 12)]
time_buckets += [(h, h+24) for h in range(72, 168, 24)]
time_buckets += [(168, float("inf"))]

################################################################################
# Utility functions for parsing ComputeCanada output into a more useful
# representation.
################################################################################
def time_difference(t1, t2, cluster="narval"):
    """Returns the amount by which time [t2] is greater than time [t1] in hours.
    Both times are expected to be in YYY-MM-DDTHH:MM:SS format. [cluster] should
    be specified, as it is needed to deal with timezone issues. By default [t1]
    is assumed to be three hours behind [t1] unless [cluster] is 'cedar'. This
    corresponds to the script being run in Vancouver.
    """
    t1 = t1.replace("-", " ").replace("T", " ").replace(":", " ").split()
    t1 = datetime(*[int(t) for t in t1])
    t1 = t1 - timedelta(hours=(0 if cluster == "cedar" else 3))
    t2 = t2.replace("-", " ").replace("T", " ").replace(":", " ").split()
    t2 = datetime(*[int(t) for t in t2])
    return (t2 - t1).seconds / 3600

def mean(x):
    if len(x) == 0:
        return 0
    else:
        return sum(x) / len(x)

def parse_memory(m):
    """Returns the amount of memory in [m] in gigabytes."""
    if m.endswith("G"):
        return int(float(m[:-1]))
    elif m.endswith("M"):
        return int(float(m[:-1])) / 1024
    elif m.endswith("T"):
        return int(float(m[:-1])) * 1024
    elif m.isdigit():
        return float(m)
    else:
        raise ValueError(f"Got weird memory string {m}")

def parse_time_limit(t):
    """Parses time limit [t], formated either as days-hours:minutes:seconds or
    hours:minutes:seconds. The time is returned as a fractional number of hours.
    """
    d = 0
    if "-" in t:
        d, t = t.split("-")
    times = t.split(":")
    if len(times) == 3:
        h, m, s = times
        return (int(d) * 24) + int(h) + (int(m) / 60) + (int(s) / 3600)
    elif len(times) == 2:
        m, s = times
        return (int(d) * 24) + (int(m) / 60) + (int(s) / 3600)
    elif len(times) == 1:
        s = times[0]
        return (int(d) * 24) + (int(s) / 3600)

def parse_resources(gpus, cpus, mem, cluster="narval"):
    """Returns a string parsing resources [r] as a usefully-formatted fraction
    of a node. This depends heavily on the cluster.
    """
    mem = parse_memory(mem)
    cpus = int(cpus)

    # If number of GPUs not specified, assume one
    if not gpus[-1].isdigit():
        gpus = f"{gpus}:1"

    try:
        if cluster == "narval":
            gpu_quarters = int(gpus[-1]) / 4
            cpu_quarters = cpus / 48
            mem_quarters = mem / 498     
            result = max(gpu_quarters, cpu_quarters, mem_quarters)
            result = int(result * 4 + .5)
            return f"A100:{result}"
        elif cluster == "cedar":
            if "v100l" in gpus:
                gpu_quarters = int(gpus[-1]) / 4
                cpu_quarters = cpus / 32
                mem_quarters = mem / 187
                result = max(gpu_quarters, cpu_quarters, mem_quarters)
                result = int(result * 4 + .5)
                return f"V100L:{result}"
            elif "gpu" in gpus:
                # In this case, we don't know what GPUs are actually being used. We
                # will assume the maximum possible resources per node without the
                # GPU type information, and log results under "GPU node"
                gpu_quarters = int(gpus[-1]) / 4
                cpu_quarters = cpus / 32
                mem_quarters = mem / 250
                result = max(gpu_quarters, cpu_quarters, mem_quarters)
                result = int(result * 4 + .5)
                return f"GPU:{result}"
            elif "p100l" in gpus:
                gpu_quarters = int(gpus[-1]) / 4
                cpu_quarters = cpus / 24
                mem_quarters = mem / 250
                result = max(gpu_quarters, cpu_quarters, mem_quarters)
                result = int(result * 4 + .5)
                return f"P100L:{result}"
            elif "p100" in gpus:
                gpu_quarters = int(gpus[-1]) / 4
                cpu_quarters = cpus / 24
                mem_quarters = mem / 125
                result = max(gpu_quarters, cpu_quarters, mem_quarters)
                result = int(result * 4 + .5)
                return f"P100:{result}"
            else:
                raise ValueError(f"Unknown configuration: gpus {gpus}")
        else:
            raise ValueError(f"Unknown cluster {cluster}")
    except:
        raise ValueError(f"Got unparsable resources: gpus {gpus} | cpus {cpus} | mem {mem}")

def parse_waits(waits):
    """Returns list of waiting times [waits] formatted."""
    return {"count": len(waits), "wait": sum(waits) / len(waits)}

################################################################################
# General utility functions. 
################################################################################
def str_to_int_key(d):
    """Returns dictionary [d] with any keys that are strings and valid digits
    mapped to integers. This is applied hierarchically. This is important
    because we use integers/floats as keys in dictionaries, but saving and
    loading these keys converts them to strings.
    """
    def int_if_possible(k):
        try:
            return int(float(k))
        except:
            return k

    if isinstance(d, dict):
        return {int_if_possible(k): str_to_int_key(v) for k,v in d.items()}
    else:
        return d

def de_default(d):
    """Returns dictionary or defaultdict [d] as a nested entirely dict, and with
    integers rounded because we commonly use this for debugging.
    """
    if isinstance(d, float):
        return round(d, 2)
    elif isinstance(d, (list, tuple)):
        return [de_default(x) for x in d]
    elif isinstance(d, dict):
        return {k: de_default(v) for k,v in d.items()} 
    else:
        return d

def merge_result_keys(x, y):
    """Merges the dictionary results [x] and [y] by averaging them using the
    count information in each.
    """
    if x.keys() == y.keys() == {"count", "wait"}:
        total_count = x["count"] + y["count"]
        total_wait = (x["wait"] * x["count"]) + (y["wait"] * y["count"])
        return {"count": total_count, "wait": total_wait / total_count}
    else:
        raise ValueError(f"{x} {y}")

def merge_cluster2lfs2resource2time2wait(x, y):
    """Returns the union of [d1] and [d2] respecting the hierarchy of each. Keys
    not mapping to dictionaries must match in their value.
    """
    if x.keys() == y.keys() == {"count", "wait"}:
        return merge_result_keys(x, y)
    else:
        intersect = {}
        for k in (x.keys() & y.keys()):
            if isinstance(x[k], dict) and isinstance(y[k], dict):
                intersect[k] = merge_cluster2lfs2resource2time2wait(x[k], y[k])
            else:
                raise ValueError()
        x_non_intersection = {k: v for k,v in x.items() if not k in y}
        y_non_intersection = {k: v for k,v in y.items() if not k in x}
        return intersect | x_non_intersection | y_non_intersection


################################################################################
# Pretty-printing functions
################################################################################

def quantize_results(level_fs2resource2time2wait, num_buckets=10):
    """Returns [level_fs2resource2result] with the LevelFS values quantized into
    [num_buckets] buckets.
    """
    bucket2resource2time2wait = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"count": 0, "wait": 0})))
    for l in sorted(level_fs2resource2time2wait):
        quantized_lfs = num_buckets * (l // num_buckets)        
        for r in sorted(level_fs2resource2time2wait[l], key=lambda x: str(x)):
            for t in sorted(level_fs2resource2time2wait[l][r]):
                quantized_time = [(s,e) for s,e in time_buckets if (s <= t and t < e)][0]

                bucket2resource2time2wait[quantized_lfs][r][quantized_time] = merge_result_keys(
                    level_fs2resource2time2wait[l][r][t],
                    bucket2resource2time2wait[quantized_lfs][r][quantized_time]
                )
                
    return bucket2resource2time2wait


def pretty_print_results(cluster2lfs2resource2time2wait, num_buckets=10):
    """Returns a string pretty-printing [cluster2_level_fs2resource2result].

    Args:
    cluster2_level_fs2resource2result -- a mapping
    num_buckets                       -- number of num_buckets to use
    """
    def get_result_str(r):
        if r["count"] > 0:
            return f"{r['wait']:.2f} ({r['count'] / 1000:.2f})"
        else:
            return ""

    s = datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "\n"
    for c,level_fs2resource2time2wait in cluster2lfs2resource2time2wait.items():

        s += f" --- {c.upper()} ---\n"
        s += " " * (13 + len(" percentile LevelFS | "))
        s += "".join([f"{f'{s}-{e}':16}" for s,e in time_buckets]) + "\n"
        s += f"{'-' * 256}\n"

        lfs2resource2time2wait = quantize_results(level_fs2resource2time2wait)

        for lfs,resource2time2wait in lfs2resource2time2wait.items():

            for r in sorted(resource2time2wait):
                
                time2wait = resource2time2wait[r]
                all_times2waits = {tuple(tb): "" for tb in time_buckets}
                all_times2waits |= {t: time2wait[t] for t in sorted(time2wait, key=lambda t: t[0])}

                assert len(all_times2waits) == len(time_buckets), all_times2waits.keys()

                s += f"{lfs:3}th percentile LevelFS: {r:6} | "
                s += "".join([f"{get_result_str(time2wait[t]):16}" for t in all_times2waits])
                s += "\n"
                
            s += f"{'-' * 256}\n"
        s += f"{'=' * 256}\n"

    return s

def get_account_to_level_fs(cluster="narval", min_level_fs=1e-5):
    """Returns the mapping from GPU accounts on [cluster] to their raw LevelFS.
    """
    def is_valid_row(row):
        return (len(row) >= 2
            and row[0].endswith("gpu"))

    print(f"{cluster.upper()}: getting sshare data... ")
    data_file = os.popen(f"ssh tme3@{cluster}.computecanada.ca '/opt/software/slurm/bin/sshare -l -o 'Account,LevelFS''")
    data = [d.split() for d in data_file.read().split("\n")]
    data = [(row[0], float(row[1])) for row in data[4:] if is_valid_row(row)]

    # For reasons I don't entirely understand, the LevelFS for my account is
    # also listed underneath the correct value for the account. We need to get
    # rid of it. We'll use the fact the sort() is stable. This is also why we
    # use a list and not a dict to store [data] initially—we need the spatial
    # co-occurenc of 'rrg-keli_gpu' and the spurious index.
    if "rrg-keli_gpu" in [d[0] for d in data]:
        correct_index = [d[0] for d in data].index("rrg-keli_gpu")
        data = data[:correct_index+1] + data[correct_index+2:]
    if "def-keli_gpu" in [d[0] for d in data]:
        correct_index = [d[0] for d in data].index("def-keli_gpu")
        data = data[:correct_index+1] + data[correct_index+2:]

    valid_level_fs = [d[1] for d in data
        if d[1] < float("inf") and d[1] >= min_level_fs]
    level_fs2percentile = {l: int(100 * idx / len(valid_level_fs))
        for idx,l in enumerate(sorted(valid_level_fs))}
    level_fs2percentile |= {l: -1 for l in [d[1] for d in data]
        if l < min_level_fs}
    level_fs2percentile |= {l: 101 for l in [d[1] for d in data]
        if l == float("inf")}

    return {a: {"percentile": level_fs2percentile[l], "value": l}
        for a,l in data}

def get_job_to_info(cluster="narval"):
    """Returns a mapping from each job pending due to the cluster not having
    infinite capacity or running on cluster [cluster] to information about the
    job we want to log.

    Notes:
    We **MUST** use 'JobArrayID' to get job IDs. Using 'JobID' and 'ArrayTaskID'
    isn't the same, as for arrays the jobID can change.
    """
    good_reasons_to_not_run = ["JobArrayTaskLimit", "QOSMaxJobsPerUserLimit"]
    
    print(f"{cluster.upper()}: getting squeue data... ")
    data_file = os.popen(f"ssh tme3@{cluster}.computecanada.ca '/opt/software/slurm/bin/squeue -r -O 'Account,JobArrayID,State:.10,EligibleTime:.30,TimeLimit:.15,tres-per-node:.30,cpus-per-task:.30,MinMemory:.20,Reason:.500''")

    curr_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    data = [d.split() for d in data_file.read().split("\n")]
    data = [d for d in data if len(d) >= 9]

    def get_job_info(account, job_id, state, time_pending,
        time_requested, gpus, cpus, mem, reason, cluster="narval"):
        """Returns a dictionary containing information about a job, or None if
        it's not a job we want to log, ie. it's not running for a good reason
        or we don't interpret its state. See the command above and the SLURM
        squeue documentation for more information on the arguments.

        Args:
        account           -- the account of the job
        job_id          -- the job's job ID. Ignored
        task_id         -- the job's task ID. Ignored
        state           -- the job's state
        time_pending    -- the time the job has been pending and not able to run
                            for a bad reason in YYYY-MM-DD-HH-MM-SS format, or
                            'N/A'
        time_requested  -- the time the job has requested (as a SLURM-formatted string)
        gpus            -- the job's requested GPUs
        cpus            -- the job's requested number of CPUs
        mem             -- the job's requested memory
        reason          -- the reason the job has for running or not running
        cluster         -- the cluster the job is run on
        """
        if state.lower() in ["running", "completing"]:
            state = "running"
        elif state.lower() == "pending":
            state = "pending"
        else:
            raise ValueError(f"Unknown state")

        if (reason in good_reasons_to_not_run
            or not "gpu" in gpus
            or time_pending == "N/A"):
            return None
        else:
            return {"account": account,
                "level_fs": "unknown",
                "state": state.lower(),
                "wait": time_difference(time_pending, curr_time, cluster=cluster),
                "time_requested": parse_time_limit(time_requested),
                "resources": parse_resources(gpus, cpus, mem, cluster=cluster)
            }

    job2info =  {d[1]: get_job_info(*d[:9], cluster=cluster)
        for d in data[1:]}
    return {j: info for j,info in job2info.items() if not info is None}
    

def get_new_data(cluster="narval", min_level_fs=1e-5):
    """Returns a dictionary giving the state of the cluster.

    Args:
    cluster         -- the cluster to get jobs on           
    """
    account2lfs = get_account_to_level_fs(cluster=cluster,
        min_level_fs=min_level_fs)
    job2info = {j: info for j,info in get_job_to_info(cluster=cluster).items()
        if info["account"] in account2lfs}
    return {"account2lfs": account2lfs, "job2info": job2info}
    
def get_info_data(account2lfs_old, job2info_old, cluster="narval", 
    max_unlogged_pending_time=.25, min_level_fs=1e-5):
    """Returns a (collected_data, updated_cluster_state) tuple. [collected_data]
    is a mapping from LevelFS percentiles to job resource configurations to a
    list of wait times.

    updated_cluster_stats   -- a dictionary equivalent to the collected
                                [account2jobs] data, but with LevelFS updates made
                                appropriately
    """
    def job2info_to_mean_wait(job2info):
        return f"{mean([info['wait'] for info in job2info.values()]):.2f} (N={len(job2info)})"

    def account_to_level_fs(a):
        """Returns the most accurate LevelFS for account [a], or raises a
        ValueError if [a] has no recorded LevelFS.
        """
        if a in account2lfs_old:
            return account2lfs_old[a]
        elif a in account2lfs_new:
            return account2lfs_new[a]
        else:
            raise ValueError(f"account {a} not in either the new or old data.")

    time_since_last_run = time_difference(cluster2info_old["last_run_time"],
        datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
        cluster=None)
    data_new = get_new_data(cluster, min_level_fs=min_level_fs)
    account2lfs_new = data_new["account2lfs"]
    job2info_new = data_new["job2info"]

    ############################################################################
    # Set the LevelFS for each job as appropriate. All jobs that exist in the
    # old data keep the LevelFS they had their. All others get the LevelFS
    # of their account as of now or the prior script run. Note that the
    # partitioning below is logically complete in the sense that each job should
    # end up in one of the eight dictionaries.
    ############################################################################
    was_pending_now_pending = {j: info | {"level_fs": job2info_old[j]["level_fs"]}
        for j,info in job2info_new.items()
        if (j in job2info_old
            and info["state"] == "pending"
            and job2info_old[j]["state"] == "pending")}
    new_now_pending = {j: info | {"level_fs": account_to_level_fs(info["account"])}
        for j,info in job2info_new.items()
        if (not j in job2info_old
            and info["state"] == "pending"
            and info["wait"] <= max_unlogged_pending_time)}
    was_pending_now_running = {j: info | {"level_fs": job2info_old[j]["level_fs"]}
        for j,info in job2info_new.items()
        if (j in job2info_old
            and info["state"] == "running"
            and job2info_old[j]["state"] == "pending")}
    new_now_running = {j: info | {"level_fs": account_to_level_fs(info["account"])}
        for j,info in job2info_new.items()
        if (not j in job2info_old
            and info["state"] == "running"
            and info["wait"] <= max_unlogged_pending_time)}

    # This is just all the jobs that were running and already logged as such. We
    # need to figure out what they are so we can make sure we're logging
    # everything correctly, but can't collect any more data from them. They need
    # to be saved so that we don't later see a still-running one of them and
    # think it's new.
    was_running_now_running = {j: info | {"level_fs": job2info_old[j]["level_fs"]}
        for j,info in job2info_new.items()
        if (j in job2info_old 
            and info["state"] == "running"
            and job2info_old[j]["state"] == "running")}

    ############################################################################
    # Jobs indicating some kind of problem.
    ############################################################################
    new_wait_too_long_justified = {j: info | {"level_fs": "unknown"}
        for j,info in job2info_new.items()
        if (not j in job2info_old
            and info["wait"] > max_unlogged_pending_time
            and info["wait"] <= time_since_last_run)}
    new_wait_too_long_unjustified = {j: info | {"level_fs": "unknown"}
        for j,info in job2info_new.items()
        if (not j in job2info_old
            and info["wait"] > max_unlogged_pending_time
            and info["wait"] > time_since_last_run)}
    was_running_now_pending = {j: info | {"level_fs": job2info_old[j]["level_fs"]}
        for j,info in job2info_new.items()
        if (j in job2info_old
            and info["state"] == "pending"
            and job2info_old[j]["state"] == "running")}

    all_updated_data = (was_pending_now_pending | new_now_pending
        | was_pending_now_running | new_now_running | was_running_now_running
        | new_wait_too_long_justified | new_wait_too_long_unjustified
        | was_running_now_pending)

    print("A", len(job2info_new))
    print("B", len(was_pending_now_pending))
    print("C", len(new_now_pending))
    print("D", len(was_pending_now_running))
    print("E", len(new_now_running))
    print("E", len(was_running_now_running))
    print("F", len(new_wait_too_long_justified))
    print("G", len(new_wait_too_long_unjustified))
    print("H", len(was_running_now_pending))

    missing_jobs = {j: info for j,info in job2info_new.items()
        if not j in all_updated_data}

    if not len(missing_jobs) == 0:
        print(f"Got missing jobs: {missing_jobs}")
    ############################################################################
    # Log data for our LevelFS to stuff to wait mapping.
    ############################################################################
    collected_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))
    for j,info in (new_now_running | was_pending_now_running).items():
        if (info["level_fs"] == "unknown"
            or info["level_fs"]["value"] == float("inf")
            or info["level_fs"]["value"] < min_level_fs):
            continue
        else:
            level_fs = info["level_fs"]["percentile"]
            resources = info["resources"]
            time = info["time_requested"]
            collected_data[level_fs][resources][time].append(info["wait"])

    ############################################################################
    # Print useful things to the screen.
    ############################################################################
    was_pending_now_pending_logged = {j: info for j,info in was_pending_now_pending.items() if not info["level_fs"] == "unknown"}
    was_pending_now_pending_unlogged = {j: info for j,info in was_pending_now_pending.items() if info["level_fs"] == "unknown"}

    new_now_running_logged = {j: info for j,info in new_now_running.items() if not info["level_fs"] == "unknown"}
    new_now_running_unlogged = {j: info for j,info in new_now_running.items() if info["level_fs"] == "unknown"}

    was_pending_now_running_logged = {j: info for j,info in was_pending_now_running.items() if not info["level_fs"] == "unknown"}
    was_pending_now_running_unlogged = {j: info for j,info in was_pending_now_running.items() if info["level_fs"] == "unknown"}

    print(f"LOG: {cluster.upper()} : all-pending {job2info_to_mean_wait({j: info for j,info in job2info_new.items() if info['state'] == 'pending'})} | all-running {job2info_to_mean_wait({j: info for j,info in job2info_new.items() if info['state'] == 'running'})}")
    print(f"LOG: {cluster.upper()} : new-now-pending {job2info_to_mean_wait(new_now_pending)} | was-pending-now-pending {job2info_to_mean_wait(was_pending_now_pending_logged)} | was-pending-now-pending (unlogged) {job2info_to_mean_wait(was_pending_now_pending_unlogged)}")
    print(f"LOG: {cluster.upper()} : new-now-running {job2info_to_mean_wait(new_now_running)} | was-pending-now-running {job2info_to_mean_wait(was_pending_now_running_logged)} | was-pending-now-running (unlogged) {job2info_to_mean_wait(was_pending_now_running_unlogged)}")
    print(f"LOG: {cluster.upper()} : was-running-now-pending {job2info_to_mean_wait(was_running_now_pending)} | wait-too-long-justified (unlogged) {job2info_to_mean_wait(new_wait_too_long_justified)} | wait-too-long-unjustified (unlogged) {job2info_to_mean_wait(new_wait_too_long_unjustified)}\n")

    print(f"CURRENTLY PENDING LOGGED DATA: {sorted(was_pending_now_pending_logged)}\n")
    print(f"JUST RAN WAS PENDING LOGGED DATA: {sorted(was_pending_now_running_logged)}\n")
    print(f"NEW WAIT TOO LONG: {sorted(new_wait_too_long_unjustified)}\n")

    data_new["job2info"] = all_updated_data
    return data_new, collected_data

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--clusters", choices=["narval", "cedar"], nargs="+",
        required=True)
    P.add_argument("--max_unlogged_pending_time", default=.3, type=float,
        help="Maximum time interval a job can have between submission and start unless it was logged as pending")
    P.add_argument("--min_level_fs", type=float, default=1e-8,
        help="Ignore accounts whose raw LevelFS is less than this. They actually could have some jobs, but very, very few, and many accounts have a LevelFS of essentially nothing; we need this to have meaningful LevelFS percentiles. Don't set to zero.")
    P.add_argument("--unknown_level_fs_okay", default=True)
    P.add_argument("--level_fs_to_wait", default=f"{os.path.dirname(__file__)}/LevelFSToWait.json",
        help="File to store the LevelFS to wait mapping for all jobs")
    P.add_argument("--human_readable", default=f"{os.path.dirname(__file__)}/HumanReadableResults.txt",
        help="File to store the LevelFS to wait mapping for all jobs")
    P.add_argument("--recent_cluster_data", default=f"{os.path.dirname(__file__)}/RecentClusterData.json",
        help="File to store recent data about the cluster to")
    args = P.parse_args()

    ############################################################################
    # Get any old data that exists.
    ############################################################################
    print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} -- min_level_fs: {args.min_level_fs} ----------")
    found_old_data = os.path.exists(args.recent_cluster_data)
    if found_old_data:
        with open(args.recent_cluster_data, "r+") as f:
            cluster2info_old = json.load(f)
        
        if os.path.exists(args.level_fs_to_wait):
            with open(args.level_fs_to_wait, "r+") as f:
                cluster2lfs2resource2time2wait_old = str_to_int_key(json.load(f))
        else:
            cluster2lfs2resource2time2wait_old = defaultdict(lambda:
                defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:
                {"count": 0, "wait": 0}))))

        cluster2info_data = {c: get_info_data(
            cluster2info_old[c]["account2lfs"],
            cluster2info_old[c]["job2info"],
            cluster=c,
            max_unlogged_pending_time=args.max_unlogged_pending_time,
            min_level_fs=args.min_level_fs)
            for c in args.clusters if c in cluster2info_old}
            
        cluster2info_new = {c: i for c,(i,_) in cluster2info_data.items()}
        cluster2info_new |= {c: get_new_data(cluster=c, min_level_fs=args.min_level_fs)
            for c in args.clusters if not c in cluster2info_old}
        cluster2info_new |= {"last_run_time": datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}

        cluster2lfs2resource2time2wait_new = {c: d
            for c,(_,d) in cluster2info_data.items()}
        cluster2lfs2resource2time2wait_new = {c: {l: {r: {t: parse_waits(waits)
            for t,waits in cluster2lfs2resource2time2wait_new[c][l][r].items()}
            for r in cluster2lfs2resource2time2wait_new[c][l]}
            for l in cluster2lfs2resource2time2wait_new[c]}
            for c in cluster2lfs2resource2time2wait_new}
        cluster2lfs2resource2time2wait = merge_cluster2lfs2resource2time2wait(
            cluster2lfs2resource2time2wait_new,
            cluster2lfs2resource2time2wait_old)

        ########################################################################
        # Write the collected data to disk
        ########################################################################
        with open(args.recent_cluster_data, "w+") as f:
            json.dump(cluster2info_new, f)
            print(f"Saved cluster2account2jobs data to {args.recent_cluster_data}")
        with open(f"{args.recent_cluster_data.replace('.json', '')}_{datetime.now().strftime('%d-%H-%M-%S')}.json", "w+") as f:
            json.dump(cluster2info_new, f)
            print(f"Saved cluster2account2jobs data to {args.recent_cluster_data}")
        with open(args.level_fs_to_wait, "w+") as f:
            json.dump(cluster2lfs2resource2time2wait, f)
            print(f"Saved cluster2lfs2resource2time2wait data to {args.level_fs_to_wait}")

        ########################################################################
        # Log data in a human-readable way
        ########################################################################
        s = pretty_print_results(cluster2lfs2resource2time2wait)
        for c in args.clusters:
            if c == "narval":
                s += f"{c.upper()} RRG LevelFS {cluster2info_new[c]['account2lfs']['rrg-keli_gpu']}\n"
            s += f"{c.upper()} DEF LevelFS: {cluster2info_new[c]['account2lfs']['def-keli_gpu']}\n"

        with open(args.human_readable, "w+") as f:
            f.write(s)
    else:
        cluster2info_old = {c: get_new_data(cluster=c, min_level_fs=args.min_level_fs)
            for c in args.clusters}
        cluster2info_old |= {"last_run_time": datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}
        with open(args.recent_cluster_data, "w+") as f:
            json.dump(cluster2info_old, f)