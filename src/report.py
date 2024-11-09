'''
    Author: University of Illinois at Urbana Champaign
    DateTime: 2023-11-05 10:53:20
    FilePath: src/report.py
    Description:
'''

import datetime
import functools
import subprocess
import time

import psutil


def date_time(t):
    return datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")


def get_file(directory: str, link_file: str):
    if 'http' not in link_file:
        return link_file
    link, file =  link_file.split(',')
    file = file.split('.')[0]
    subprocess.run(f'gdown --fuzzy {link}', shell=True)
    subprocess.run(f'unzip -o {file}.zip -d {directory}/', shell=True)
    return f'{directory}/{file}.txt'


def _report(func):
    p = psutil.Process()
    with p.oneshot():
        name = p.name()
        cmdline = ' '.join(p.cmdline())
        create_time = date_time(p.create_time())
        cpu_times = p.cpu_times()
        cpu_total = sum(cpu_times)
        memory = p.memory_info()
        ram_total = memory.rss / 1024 / 1024 / 1024
        num_threads = p.num_threads()
        print(f'{create_time}: {cmdline}')
        print(f'{func.__name__}: process [{name}], number of threads',
            f'{num_threads}.')
        print(f'{func.__name__}: total CPU {cpu_total:.3f} seconds,',
            f'total memory {ram_total:.3f} GB.')


def report(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        _report(func)
        print(f'{func.__name__}: finishes in {elapsed_time:.3f} seconds.')
        return result
    return new_func
