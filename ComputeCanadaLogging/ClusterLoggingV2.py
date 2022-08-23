#!/usr/bin/env python3
"""Script to collect data for another part of the ComputeCanada LevelFS
understanding saga!

When this script is run at time [t], it will
(a) save a JSON file that says exactly what our group is doing on various
    ComputeCanada clusters. This JSON file contains a key for each cluster. Each
    such key maps to a dictionary with two keys. The first key maps to a
    dictionary where each key is for an account, either `def-keli_gpu` o
     'rrg-keli_gpu' on the cluster. The second is 'job2task2info', which maps to
     such a dictionary.

(b) look through jobs (not tasks) that have recently completed, and if they were
    one-off or sequential array jobs, log a (LevelFS, job resources/time, wait)
    datapoint for each one-off job or task in the array.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timedelta
import os
import time
import json
import random
from multiprocessing import Pool
from functools import partial

################################################################################
# Global scope contant-y things.
################################################################################
time_buckets = [(h, h+3) for h in range(0, 12, 3)] + [[12,18], [18,24]]
time_buckets += [(h, h+12) for h in range(24, 72, 12)]
time_buckets += [(h, h+24) for h in range(72, 168, 24)]
time_buckets += [(168, float("inf"))]

################################################################################
# Utility functions for parsing ComputeCanada output into a more useful
# representation.
################################################################################
def time_to_str(t): return t.strftime("%Y-%m-%d %H:%M:%S")

def subtract_three_hours(t):
    """Returns string [t] in YYYY-MM-DDTHH:MM:SS format with three hours
    subtracted.
    """
    return time_to_str(to_datetime(t) - timedelta(hours=3))

def to_datetime(t):
    """Returns string [t] in YYY-MM-DDTHH:MM:SS format as a datetime object."""
    t = t.replace("-", " ").replace("T", " ").replace(":", " ").split()
    return datetime(*[int(t_) for t_ in t])

def time_difference(t1, t2, cluster="narval"):
    """Returns the amount by which time [t2] is greater than time [t1] in hours.
    Both times are expected to be in YYY-MM-DDTHH:MM:SS format. [cluster] should
    be specified, as it is needed to deal with timezone issues. By default [t1]
    is assumed to be three hours behind [t1] unless [cluster] is 'cedar'. This
    corresponds to the script being run in Vancouver.
    """
    t1, t2 = to_datetime(t1), to_datetime(t2)
    t1 = t1 - timedelta(hours=(0 if cluster == "narval" else 3))
    return (t2 - t1).total_seconds() / 3600

def mean(x): return 0 if len(x) == 0 else (sum(x) / len(x))

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

def get_job_parallelism(job_id, cluster):
    """Returns if job [job_id] is parallel or not as an integer."""
    job_id = job_id[:job_id.rindex("_")] if "_" in job_id else job_id
    r = os.popen(f"ssh tme3@{cluster}.computecanada.ca '/opt/software/slurm/bin/scontrol show job {job_id}'").read()
    if "ArrayTaskThrottle=" in r:
        start = r.rindex("ArrayTaskThrottle=") + len("ArrayTaskThrottle=")
        end = r[start:].index(" ") + start
        return int(r[start:end])
    else:
        return 1

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

def get_job_to_task_to_info(cluster="narval", max_wait=.3, accounts=["rrg-keli_gpu", "def-keli_gpu"]):
    """Returns a mapping from each job pending due to the cluster not having
    infinite capacity or running on cluster [cluster] to its tasks to
    information about the state of each task. One-off jobs (ie. ones that aren't
    arrays, are treated as arrays of size one.)

    Notes:
    We **MUST** use 'JobArrayID' to get job IDs. Using 'JobID' and 'ArrayTaskID'
    isn't the same, as for arrays the jobID can change.
    """
    print(f"{cluster.upper()}: getting squeue data... ")
    data_file = os.popen(f"ssh tme3@{cluster}.computecanada.ca '/opt/software/slurm/bin/squeue -r -A rrg-keli_gpu -O   'JobArrayID,Account,ArrayTaskID,State:.10,EligibleTime:.30,TimeLimit:.15,tres-per-node:.30,cpus-per-task:.30,MinMemory:.20,Reason:.500''")

    curr_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    data = [d.split() for d in data_file.read().split("\n")]
    data = [d for d in data if len(d) >= 9]

    def get_job_info(job_id, account, task_id, state, submit_time,
        time_requested, gpus, cpus, mem, cluster="narval"):
        """Returns a dictionary containing information about a job, or None if
        it's not a job we want to log, ie. it's not running for a good reason
        or we don't interpret its state. See the command above and the SLURM
        squeue documentation for more information on the arguments.

        Args:
        account           -- the account of the job
        job_id          -- the job's job ID. Ignored
        task_id         -- the job's task ID
        state           -- the job's state
        submit_time     -- the time the job was submitted or 'N/A'
        time_requested  -- the time the job has requested (as a SLURM-formatted string)
        gpus            -- the job's requested GPUs
        cpus            -- the job's requested number of CPUs
        mem             -- the job's requested memory
        cluster         -- the cluster the job is run on
        """
        if state.lower() in ["running", "completing"]:
            state = "running"
        elif state.lower() == "pending":
            state = "pending"
        else:
            raise ValueError(f"Unknown state")

        if not "gpu" in gpus or not account in accounts:
            return None
        else:
            return {
                "task_id": 0 if task_id == "N/A" else int(task_id),
                "account": account,
                "state": state.lower(),
                "level_fs": "unknown",
                "eligible": "no",
                "submitted": "N/A" if submit_time == "N/A" else subtract_three_hours(submit_time),
                "time_requested": parse_time_limit(time_requested),
                "resources": parse_resources(gpus, cpus, mem, cluster=cluster),
                "parallel": "unknown",
            }

    job2info =  {d[0]: get_job_info(*d[:9], cluster=cluster)
        for d in data[1:]}
    job2info = {j: info for j,info in job2info.items() if not info is None}
    jobs = {(j[:j.rindex("_")] if "_" in j else j) for j in job2info}
    job2task2info = {j: {t: info for t,info in job2info.items() if t.startswith(j)}
        for j in jobs}

    with Pool(processes=10) as p:
        jobs = list(job2task2info.keys())
        parallels = p.map(partial(get_job_parallelism, cluster=cluster), jobs)
        job2parallel = {j: p for j,p in zip(jobs, parallels)}

    job2task2info = {j: {t: info | {"parallel": job2parallel[j]}
        for t,info in task2info.items()}
        for j,task2info in job2task2info.items()}

    ############################################################################
    # Eligible jobs are jobs that can have their LevelFS recorded, but this
    # LevelFS should be overwritten with the known LevelFS values logged for the
    # job previously if it exists. Here, we mark eligible jobs with the time
    # they became eligible. Jobs are eligible if either
    # (a) they are running and were submitted a short time ago. In this case, we
    #   record the time they were submitted
    # (b) they are tasks of a job array that can run [p] jobs at once, [r] tasks
    #   are running, and they are the [p - r] next pending jobs in the array. We
    #   record the current time.
    ############################################################################

    def eligible_if_is_running_and_submitted_recently(info, max_wait, cluster):
        """Returns the Eligible of job information [info] of job [job] in
        [job2task2info].
        """
        if not info["submitted"] == "N/A" and info["state"] == "running":
            curr_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            wait = time_difference(info["submitted"], curr_time, cluster=cluster)
            return info["submitted"] if wait < max_wait else info["eligible"]
        else:
            return info["eligible"]

    job2task2info = {j: {t: info | {"eligible": eligible_if_is_running_and_submitted_recently(info, max_wait, cluster)}
        for t,info in task2info.items()}
        for j,task2info in job2task2info.items()}

    def is_pending_but_not_throttled(job, info, job2task2info):
        """Returns the Eligible of job information [info] of job [job] in
        [job2task2info].
        """
        job_parallelism = list(job2task2info[job].values())[0]["parallel"]
        running_tasks = [info for info in job2task2info[job].values()
            if info["state"] == "running"]
        pending_task_ids = [info["task_id"]
            for info in job2task2info[job].values()
            if info["state"] == "pending"]

        num_could_run = job_parallelism - len(running_tasks)
        cutoff = min(len(pending_task_ids), num_could_run)
        if (not info["submitted"] == "N/A"
            and info["task_id"] in pending_task_ids[:cutoff]):
            return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        else:
            return info["eligible"]

    job2task2info = {j: {t: info | {"eligible": is_pending_but_not_throttled(j, info, job2task2info)}
        for t,info in task2info.items()}
        for j,task2info in job2task2info.items()}

    return job2task2info

def get_cluster_state(cluster="narval", accounts=["rrg-keli_gpu", "def-keli_gpu"],
    min_level_fs=1e-3,
    max_wait=.35):
    """Returns a dictionary giving the state of account [account] on cluster
    [cluster].

    Args:
    account         -- the account to get data for
    cluster         -- the cluster to get data on
    min_level_fs    -- minimum LevelFS for percentile computation
    """
    # Get the raw and percentile LevelFS
    account2lfs = {a: l
        for a,l in get_account_to_level_fs(cluster=cluster,
            min_level_fs=min_level_fs).items()
        if a in accounts}

    current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    job2task2info = get_job_to_task_to_info(cluster=cluster, max_wait=max_wait)
    job2task2info = {j: task2info for j,task2info in job2task2info.items()
        if all([info["account"] in accounts
            for info in task2info.values()])}

    # Log LevelFS values for all jobs for which the 'eligible' key isn't 'no'.
    # If older LevelFS values exist, we will switch to them later.
    job2task2info = {j: {t: info | {"level_fs": account2lfs[info["account"]] if not info["eligible"] == "no" else "unknown"}
        for t,info in task2info.items()}
        for j,task2info in job2task2info.items()}

    return {"job2task2info": job2task2info, "account2level_fs": account2lfs}


if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--clusters", choices=["narval", "cedar"], nargs="+",
        required=True,
        help="Clusters for which to get data")
    P.add_argument("--max_wait", default=.3, type=float,
        help="Maximum time interval a job can have between submission and start unless it was logged as pending")
    P.add_argument("--min_level_fs", type=float, default=1e-8,
        help="Ignore accounts whose raw LevelFS is less than this. They actually could have some jobs, but very, very few, and many accounts have a LevelFS of essentially nothing; we need this to have meaningful LevelFS percentiles. Don't set to zero.")
    P.add_argument("--level_fs_to_wait", default=f"{os.path.dirname(__file__)}/LevelFSToWait.json",
        help="File to store the LevelFS to wait mapping for all jobs")
    P.add_argument("--human_readable", default=f"{os.path.dirname(__file__)}/HumanReadableResults.txt",
        help="File to store the LevelFS to wait mapping for all jobs")
    P.add_argument("--recent_cluster_data", default=f"{os.path.dirname(__file__)}/RecentClusterData.json",
        help="File to store recent data about the cluster to")
    P.add_argument("--data_storage_folder", default=f"{os.path.abspath(os.path.dirname(__file__))}/DataStorage",
        help="Folder to store data in")
    args = P.parse_args()

    print(f"=========== {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} -- LevelFS below which accounts are ignored for percentile computation: {args.min_level_fs} ===========")

    ############################################################################
    # Get the current state of the cluster. Jobs that have been
    # submitted recently and are able to run but for priority are marked as such
    # with 'log_level_fs' and their LevelFS is logged. Otherwise, LevelFS
    # is set to 'unknown'.
    ############################################################################
    cluster2acount2lfs_job2task2info = {c: get_cluster_state(cluster=c,
        min_level_fs=args.min_level_fs,
        max_wait=args.max_wait)
        for c in args.clusters}

    ############################################################################
    # Load the last collected data. Any known LevelFS values in the past data
    # overwrite those in the current data, as do values of 'eligible' if they
    # aren't 'no'.
    ############################################################################
    if os.path.exists(f"{args.data_storage_folder}/MostRecentData.json"):
        with open(f"{args.data_storage_folder}/MostRecentData.json", "r+") as f:
            cluster2acount2lfs_job2task2info_old = json.load(f)
            cluster2job2task2info_old = {c: v["job2task2info"]
                for c,v in cluster2acount2lfs_job2task2info_old.items()}
    else:
        cluster2job2task2info_old = {c: {"job2task2info": {}} for c in args.clusters}

    def get_level_fs(cluster, job, task, info, cluster2job2task2info_old):
        """Returns the LevelFS value to use in [info] of [task] of [job] of
        [cluster] in current data given previous data
        [cluster2job2task2info_old].
        """
        if (cluster in cluster2job2task2info_old
            and job in cluster2job2task2info_old[cluster]
            and task in cluster2job2task2info_old[cluster][job]
            and not cluster2job2task2info_old[cluster][job][task]["level_fs"] == "unknown"):
            return cluster2job2task2info_old[cluster][job][task]["level_fs"]
        else:
            return info["level_fs"]

    def get_eligible(cluster, job, task, info, cluster2job2task2info_old):
        """Returns the Eligible value to use in [info] of [task] of [job] of
        [cluster] in current data given previous data
        [cluster2job2task2info_old].
        """
        if (cluster in cluster2job2task2info_old
            and job in cluster2job2task2info_old[cluster]
            and task in cluster2job2task2info_old[cluster][job]
            and not cluster2job2task2info_old[cluster][job][task]["eligible"] in ["no", "already_logged"]):
            return cluster2job2task2info_old[cluster][job][task]["eligible"]
        else:
            return info["eligible"]


    cluster2job2task2info = {c: {j: {t: info | {
            "level_fs": get_level_fs(c, j, t, info, cluster2job2task2info_old),
            "eligible": get_eligible(c, j, t, info, cluster2job2task2info_old)}
        for t,info in task2info.items()}
        for j,task2info in acount2lfs_job2task2info["job2task2info"].items()}
        for c,acount2lfs_job2task2info in cluster2acount2lfs_job2task2info.items()}

    ############################################################################
    # Infer data about what the cluster does from the now-updated current data.
    # When logging data from running jobs, set their values of Eligible to
    # 'already_logged' so they are not logged again. This is done in the
    # placeholder [cluster2job2task2info]; [cluster2acount2lfs_job2task2info] is
    # updated from the placeholder next.
    ############################################################################
    curr_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    cluster2lfs2resource2time2waits = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: []))))

    for c,job2task2info in cluster2job2task2info.items():
        for j in cluster2job2task2info[c]:
            for t,info in cluster2job2task2info[c][j].items():
                if (info["state"] == "running"
                    and not info['eligible'] in ["no", "already_logged"]
                    and not info["level_fs"] == "unknown"
                    and time_difference(info["eligible"], curr_time) < args.max_wait):
                    wait = time_difference(info["eligible"], curr_time)
                    cluster2lfs2resource2time2waits[c][info["level_fs"]["percentile"]][info["resources"]][info["time_requested"]].append(wait)
                    cluster2job2task2info[c][j][t]["eligible"] = "already_logged"
                    print(f"LOG: LevelFS {info['level_fs']['percentile']} for with wait {wait} logged for recently submitted job/task {t}")
                elif (info["state"] == "running"
                    and c in cluster2job2task2info_old
                    and j in cluster2job2task2info_old[c]
                    and t in cluster2job2task2info_old[c][j]
                    and cluster2job2task2info_old[c][j][t]["state"] == "pending"
                    and not info['eligible'] in ["no", "already_logged"]
                    and not info["level_fs"] == "unknown"):
                    wait = time_difference(info["eligible"], curr_time)
                    cluster2lfs2resource2time2waits[c][info["level_fs"]["percentile"]][info["resources"]][info["time_requested"]].append(wait)
                    cluster2job2task2info[c][j][t]["eligible"] = "already_logged"
                    print(f"LOG: LevelFS {info['level_fs']['percentile']} for with wait {wait} logged for previously pending job/task {t}")
                else:
                    continue
        else:
            continue

    cluster2acount2lfs_job2task2info = {c: acount2lfs_job2task2info | {"job2task2info": cluster2job2task2info[c]}
        for c,acount2lfs_job2task2info in cluster2acount2lfs_job2task2info.items()}

    cluster2lfs2resource2time2wait = {c: {lfs: {r: {t: {"count": len(wait), "wait": mean(wait)}
        for t,wait in time2wait.items()}
        for r,time2wait in resource2time2wait.items()}
        for lfs,resource2time2wait in lfs2resource2time2wait.items()}
        for c,lfs2resource2time2wait in cluster2lfs2resource2time2waits.items()}

    ############################################################################
    # Combine the existing LevelFS data with the old accumulated LevelFS data
    ############################################################################
    if os.path.exists(f"{args.data_storage_folder}/LevelFSToWait.json"):
        with open(f"{args.data_storage_folder}/LevelFSToWait.json", "r+") as f:
            cluster2lfs2resource2time2wait_old = str_to_int_key(json.load(f))
    else:
        cluster2lfs2resource2time2wait_old = {}
    cluster2lfs2resource2time2wait = merge_cluster2lfs2resource2time2wait(
        cluster2lfs2resource2time2wait,
        cluster2lfs2resource2time2wait_old)

    ############################################################################
    # Write the updated data to disk
    ############################################################################
    if not os.path.exists(args.data_storage_folder):
        os.makedirs(args.data_storage_folder)
    with open(f"{args.data_storage_folder}/cluster2data_{datetime.now().strftime('%y-%m-%d-%H-%M')}.json", "w+") as f:
        json.dump(cluster2acount2lfs_job2task2info, f)
    with open(f"{args.data_storage_folder}/LevelFSToWait.json", "w+") as f:
        json.dump(cluster2lfs2resource2time2wait, f)
    with open(f"{args.data_storage_folder}/MostRecentData.json", "w+") as f:
        json.dump(cluster2acount2lfs_job2task2info, f)

    ############################################################################
    # Write nice human-readable results
    ############################################################################
    human_readable_results = pretty_print_results(cluster2lfs2resource2time2wait, num_buckets=10)
    level_fs_data = {c: cluster2acount2lfs_job2task2info[c]["account2level_fs"] for c in args.clusters}
    print(human_readable_results)
    print(level_fs_data)

    with open(f"{os.path.dirname(os.path.abspath(__file__))}/HumanReadableResults.txt", "w+") as f:
        f.write(human_readable_results)
