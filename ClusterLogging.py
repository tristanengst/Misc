#!/usr/bin/env python3

"""Script to log things from the cluster. Should be run every 10 minutes or so.
It maintains three dicitonaries as JSON files giving the data we care about.

The basic idea is to frequently log the LevelFS of each group, the jobs that
are pending, and the jobs that are running. This allows us to construct a
per-cluster from group LevelFS percentiles when jobs were submitted to the
resources requested for the job (measured in quarters of a node) to the amount
of time requested for the job (measured in quantized hours) to the average time
it took for the job to start. We can use this mapping to figure out what LevelFS
percentile we want to have.

Args:
All arguments are set with intelligent defaults.

--max_unlogged_pending  : the maximum time a job could have been pending for us
                            to log it with the last recorded LevelFS for its
                            group in the case where we never see it pending and
                            thus don't have a LevelFS to log for when it was
                            submitted. This should be the same as the interval
                            this script is run at.
--min_level_fs          : the minimum (raw, not percentile) LevelFS we should
                            log data for. We think groups with very small
                            LevelFS values might get jobs very fast, in
                            contradiction to the stated policies
--level_fs_to_wait      : JSON file to write the mapping described above to
--recent_cluster_data   : JSON file logging recent cluster data
--human_readable        : human-readable script outputs as a .txt file
"""
import argparse
from collections import defaultdict
from datetime import datetime
import os
import time
import json

################################################################################
# Global scope contant-y things.
################################################################################
clusters = ["narval", "cedar"]

time_buckets = [(h, h+3) for h in range(0, 12, 3)] + [[12,18], [18,24]]
time_buckets += [(h, h+12) for h in range(24, 72, 12)]
time_buckets += [(h, h+24) for h in range(72, 168, 24)]
time_buckets += [(168, float("inf"))]

################################################################################
# Utility functions for parsing ComputeCanada output into a more useful
# representation.
################################################################################
def parse_memory(m):
    """Returns the amount of memory in [m] in gigabytes."""
    if m.endswith("G"):
        return int(float(m[:-1]))
    elif m.endswith("M"):
        return int(float(m[:-1])) / 1024
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
    """Returns dictionary or defaultdict [d] as a nested entirely dict."""
    return {k: de_default(v) for k,v in d.items()} if isinstance(d, dict) else d

def merge_result_keys(x, y):
    """Merges the dictionary results [x] and [y] by averaging them using the
    count information in each.
    """
    if x.keys() == y.keys() == {"count", "wait"}:
        total_count = x["count"] + y["count"]
        total_wait = x["wait"] * x["count"] + y["wait"] * y["count"]
        return {"count": total_count, "wait": total_wait / total_count}
    else:
        raise ValueError(f"{x} {y}")

def merge_cluster2level_fs2resource2wait(x, y):
    """Returns the union of [d1] and [d2] respecting the hierarchy of each. Keys
    not mapping to dictionaries must match in their value.
    """
    if x.keys() == y.keys() == {"count", "wait"}:
        return merge_result_keys(x, y)
    else:
        intersect = {}
        for k in (x.keys() & y.keys()):
            if isinstance(x[k], dict) and isinstance(y[k], dict):
                intersect[k] = merge_cluster2level_fs2resource2wait(x[k], y[k])
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


def pretty_print_results(cluster2level_fs2resource2time2wait, num_buckets=10):
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
    for c,level_fs2resource2time2wait in cluster2level_fs2resource2time2wait.items():

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

def get_group_to_level_fs(cluster="narval",  min_level_fs=1e-5):
    """Returns the mapping from groups on [cluster] to LevelFS percentiles."""
    def is_valid_row(row):
        return (len(row) >= 2
            and row[0].endswith("gpu")
            and float(row[1]) >= min_level_fs
            and float(row[1]) < float("inf"))

    print(f"{cluster.upper()}: getting sshare data... ")
    data_file = os.popen(f"ssh tme3@{cluster}.computecanada.ca '/opt/software/slurm/bin/sshare -l -o 'Account,LevelFS''")
    data = [d.split() for d in data_file.read().split("\n")]
    data = [(row[0], float(row[1])) for row in data[4:] if is_valid_row(row)]

    # For reasons I don't entirely understand, the LevelFS for my account is
    # also listed underneath the correct value for the group. We need to get
    # rid of it. We'll use the fact the sort() is stable.
    if "rrg-keli_gpu" in [d[0] for d in data]:
        correct_index = [d[0] for d in data].index("rrg-keli_gpu")
        data = data[:correct_index+1] + data[correct_index+2:]
    correct_index = [d[0] for d in data].index("def-keli_gpu")
    data = data[:correct_index+1] + data[correct_index+2:]

    data = sorted(data, key=lambda x: x[1])
    level_fs_values = [d[1] for d in data]
    level_fs2percentile = {l: int(100 * idx / len(level_fs_values))
        for idx,l in enumerate(sorted(level_fs_values))}

    return {g: level_fs2percentile[l] for g,l in data}

def get_group_to_job_info(cluster="narval"):
    """Returns a mapping from each group with running jobs on cluster [cluster]
    to a dictionary
    {
        "pending": {job_id: {"resources": resources} ... },
        "running": {job_id: {"resources": resources, "wait": wait} ... }
    }

    where [resources] are the resources requested for the job and [wait] is the
    time the job waited to be run.
    """
    good_reasons_to_not_run = ["JobArrayTaskLimit", "QOSMaxJobsPerUserLimit"]
    print(f"{cluster.upper()}: getting squeue data... ")
    data_file = os.popen(f"ssh tme3@{cluster}.computecanada.ca '/opt/software/slurm/bin/squeue -r -O 'Account,JobID,ArrayTaskID,State:.10,PendingTime:.15,TimeLimit:.15,tres-per-node:.30,cpus-per-task:.30,MinMemory:.20,Reason:.500''")
    data = [d.split() for d in data_file.read().split("\n")]
    data = [d for d in data if len(d) >= 10]

    group2jobs = defaultdict(lambda: {"running": {}, "pending": {}})
    for data_item in data[1:]:
        group,job_id,task_id,state,time_pending,time_requested,gpus,cpus,mem,reason = data_item[:10]
        job_id = job_id if task_id == "N/A" else f"{job_id}_{task_id}"

        if (gpus.startswith("gres")
            and state == "PENDING"
            and not reason in good_reasons_to_not_run):
            group2jobs[group]["pending"] |= {job_id: {
                "resources": parse_resources(gpus, cpus, mem, cluster=cluster),
                "time_requested": parse_time_limit(time_requested),
            }}
        elif gpus.startswith("gres") and state == "RUNNING":
            group2jobs[group]["running"] |= {job_id: {
                "wait": int(time_pending) / 3600,
                "resources": parse_resources(gpus, cpus, mem, cluster=cluster),
                "time_requested": parse_time_limit(time_requested),
            }}
        else:
            # Ignore non-GPU jobs
            continue
        
    return group2jobs

def get_group_to_data(cluster="narval", min_level_fs=1e-5):
    """Returns a mapping from each group on cluster [cluster] to a dictionary
    {
        "level_fs": level_fs,
        "pending": {job_id: {"resources": resources, "level_fs": level_fs} ... }
        "running": {job_id: {"resources": resources, "wait": wait} ... }
    }

    where [resources] are the resources requested for the job,
    [lfs is the LevelFS percentile of the group, and [wait] is
    the average time the job waited to be run.
    """
    group2level_fs = get_group_to_level_fs(cluster=cluster,
        min_level_fs=min_level_fs)
    group2jobs = get_group_to_job_info(cluster=cluster)
    return {g: {"level_fs": l} | group2jobs[g] for g,l in group2level_fs.items()}
    
def get_group_to_new_data(group2jobs_old={}, cluster="narval", 
    max_unlogged_pending=.25, min_level_fs=1e-5):
    """Returns a (collected_data, updated_cluster_state) tuple. [collected_data]
    is a mapping from LevelFS percentiles to job resource configurations to a
    list of wait times.
    """
    group2jobs_new = get_group_to_data(cluster, min_level_fs=min_level_fs)

    collected_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))
    ############################################################################
    # Find data to log. 
    ############################################################################
    for g in group2jobs_new:
        # If this is the first time we've seen the group, we've no past LevelFS
        # data and should just ignore it this time.
        if not g in group2jobs_old:
            continue
            
        # Jobs that we logged a LevelFS for in the prior run and are now
        # running. We already assumed a LevelFS for them on a previous run.
        was_pending_now_running = {j: info
            for j,info in group2jobs_new[g]["running"].items()
            if j in group2jobs_old[g]["pending"] and "level_fs" in info}

        # Jobs that are now running that we never logged as pending because they
        # started too quickly. We assume the prior LevelFS of the group, but
        # only if the job was pending for less than a [max_unlogged_pending]
        # fraction of an hour. Otherwise, it was pending for a long time and we
        # never logged it because this script wasn't collecting results that far
        # back in time.
        level_fs = group2jobs_old[g]["level_fs"]
        never_pending_now_running = {j: info | {"level_fs": level_fs}
            for j,info in group2jobs_new[g]["running"].items()
            if (not j in group2jobs_old[g]["pending"]
                and not "level_fs" in info
                and info["wait"] < max_unlogged_pending)}

        new_running_jobs = was_pending_now_running | never_pending_now_running
        for r,info in new_running_jobs.items():
            level_fs = info["level_fs"]
            resources = info["resources"]
            req_time = info["time_requested"]
            collected_data[level_fs][resources][req_time].append(info["wait"])
        
        group2jobs_new[g]["running"] |= new_running_jobs

    collected_data = {d: dict(v) for d,v in collected_data.items()}
    ############################################################################
    # Update [group2jobs_new].
    # (a) currently pending jobs that were pending in the old data keep the
    #   recorded LevelFS of the job in the old data
    # (b) newly pending jobs get the LevelFS of the group in the old data. This
    #   will slightly underestimate it, but will never dramatically
    #   underestimate it as could happen if we used the current LevelFS.
    #   (Consider: (1) group submits lots of jobs, (2) group LevelFS tanks, (3)
    #   this script runs. The LevelFS logged prior to (1) is the best estimate
    #   of the one used to run the jobs.)
    # (c) Running jobs all have a LevelFS key. This was accomplished above.
    ############################################################################
    for g in group2jobs_new:
        # Again, because we log data from the previous iteration, we ignore
        # groups the first time we see them.
        if not g in group2jobs_old:
            continue

        # The (|) operator gives precedence to the right hand input, so any old
        # LevelFS values are preserved. This ensures that every pending job will
        # have a LevelFS key.
        level_fs = group2jobs_old[g]["level_fs"]
        group2jobs_new[g]["pending"] = {j: {"level_fs": level_fs} | info
            for j,info in group2jobs_new[g]["pending"].items()}
    
    return collected_data, group2jobs_new

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--max_unlogged_pending", default=.25, type=float,
        help="Maximum time interval a job can have between submission and start unless it was logged as pending")
    P.add_argument("--min_level_fs", type=float, default=1e-5,
        help="Ignore groups whose raw LevelFS is less than this")
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
    print("--------------------------------------------------------------")
    found_old_data = os.path.exists(args.recent_cluster_data)
    if found_old_data:
        with open(args.recent_cluster_data, "r+") as f:
            cluster2group2jobs = json.load(f)
    else:
        cluster2group2jobs = {c: get_group_to_data(cluster=c) for c in clusters}

    ############################################################################
    # If the old cluster data didn't exist, we can' do anything but wait to run
    # the script again, so we write it to disk and abort. Otherwise, we can
    # continue.
    ############################################################################
    if not found_old_data:
        with open(args.recent_cluster_data, "w+") as f:
            json.dump(cluster2group2jobs, f)
    else:
        # Load the file we write LevelFS and wait time data to if it exists,
        # otherwise load a blank template for it
        if os.path.exists(args.level_fs_to_wait):
            with open(args.level_fs_to_wait, "r+") as f:
                cluster2level_fs2resource2time2wait = str_to_int_key(json.load(f))
        else:
            cluster2level_fs2resource2time2wait = defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(
                            lambda: {"count": 0, "wait": 0}))))
        
        cluster2level_fs2resource2time2wait_new = {}
        for c in clusters:

            if not c in cluster2group2jobs:
                cluster2group2jobs[c] = get_group_to_data(cluster=c)

            level_fs2resource2time2waits, group2jobs = get_group_to_new_data(
                group2jobs_old=cluster2group2jobs[c],
                cluster=c,
                max_unlogged_pending=args.max_unlogged_pending,
                min_level_fs=args.min_level_fs)
        
            level_fs2resource2time2wait = {l: {r: {t: parse_waits(waits)
                for t,waits in level_fs2resource2time2waits[l][r].items()}
                for r in level_fs2resource2time2waits[l]}
                for l in level_fs2resource2time2waits}
            
            cluster2group2jobs[c] = de_default(group2jobs)
            cluster2level_fs2resource2time2wait_new[c] = level_fs2resource2time2wait

        cluster2level_fs2resource2time2wait = merge_cluster2level_fs2resource2wait(
            cluster2level_fs2resource2time2wait_new,
            cluster2level_fs2resource2time2wait)

        print(pretty_print_results(cluster2level_fs2resource2time2wait))
        print(f"NARVAL RRG LevelFS: {cluster2group2jobs['narval']['rrg-keli_gpu']['level_fs']}")
        print(f"NARVAL DEF LevelFS: {cluster2group2jobs['narval']['def-keli_gpu']['level_fs']}")
        # print(f"CEDAR DEF LevelFS: {cluster2group2jobs['cedar']['def-keli_gpu']['level_fs']}")
        print("------------------------------------")
        
        ########################################################################
        # Write the collected data to disk
        ########################################################################
        with open(args.recent_cluster_data, "w+") as f:
            json.dump(cluster2group2jobs, f)
        with open(args.level_fs_to_wait, "w+") as f:
            json.dump(cluster2level_fs2resource2time2wait, f)

        ########################################################################
        # Log data in a human-readable way
        ########################################################################
        s = pretty_print_results(cluster2level_fs2resource2time2wait)
        s += f"NARVAL RRG LevelFS: {cluster2group2jobs['narval']['rrg-keli_gpu']['level_fs']}\n"
        s += f"NARVAL DEF LevelFS: {cluster2group2jobs['narval']['def-keli_gpu']['level_fs']}\n"
        # s += f"CEDAR DEF LevelFS: {cluster2group2jobs['cedar']['def-keli_gpu']['level_fs']}\n"

        with open(args.human_readable, "w+") as f:
            f.write(s)
