import os.path, io, sys, gzip
import re, cv2
import math
import datetime as dt

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots

from pprint import pprint

RESULTS_DIR = os.path.join(os.getenv('RESULTS_ROOT'), 'surface_graph', 'results')
OUTPUT_DIR = os.path.join(os.getenv("RESULTS_ROOT"), 'surface_graph')

class ExpData:
    def __init__(self):
        self.pr_vector: list = []
        self.benchmark: str = ''
        self.output_file: str = ''
        self.log_file: str = ''
        self.heat_file: str =  ''
        self.reference_file: str = ''
        self.hb_df = pd.DataFrame()

app_map = { 0:'blackscholes',
            1:'bodytrack',
            2:'canneal',
            3:'streamcluster',
            4:'swaptions',
            5:'x264'
        }

def app_mapping(path):
    file = open(path)
    id = 0
    for line in file.readlines():
        id = line.split(',')[1]
        break

    return app_map[id]


PIXEL_MAX = 255.0
def psnr(original, contrast):
    mse = np.mean((original - contrast) ** 2)

    if mse == 0:
        return 100
    
    psnr = 20 * math.log10(PIXEL_MAX/ math.sqrt(mse))
    return psnr


def parse_images(video_path: str) -> list:
    encoded_vid = cv2.VideoCapture(video_path)
            
    frames = []
    succ, img = encoded_vid.read()
    while succ:
        frames.append(img) 
        succ, img = encoded_vid.read()

    return frames


def compile_testdata(label: str, path, benchmark:str = None) -> list[ExpData]:
    experiment_data = []

    for dirname in sorted(os.listdir(path)):
        data_path = ''

        if label not in dirname:
            continue
        
        data_path = os.path.join(path, dirname)
        data: ExpData = ExpData()           
        
        # save reference to output file and log_file
        for file in sorted(os.listdir(data_path)):
            if 'appmapping.txt' in file:
                map_file = os.path.join(data_path, file)
                data.benchmark = app_mapping(map_file)

            if 'output.txt' in file:
                data.output_file = os.path.join(data_path, file)
            elif 'poses.txt' in file:
                data.output_file = os.path.join(data_path, file)
            elif '.264' in file: 
                data.output_file = os.path.join(data_path, file)


            if 'execution.log.gz' in file:
                data.log_file = os.path.join(data_path, file)

            if 'PeriodicThermal.log.gz' in file:
                data.heat_file = os.path.join(data_path, file)
                
            if 'hb.log' in file:
                file_df = pd.read_csv(os.path.join(data_path, file), sep='\t')

                data.hb_df = pd.concat([data.hb_df, file_df])
            
            if benchmark:
                data.benchmark = benchmark

            data.pr_vector = [e for e in data_path.split(':')[1].split(',')]
            
        experiment_data.append(data)

    return experiment_data 

def get_benchmark_output(experiment_data : ExpData):
    run_out = []

    # TODO: not finished needs to be vector difference.
    if experiment_data.benchmark in {'bodytrack', 'blackscholes'}:
        file =  open(experiment_data.output_file, 'r')
        for line in file.readlines():
            run_out += [float(num) for num in  re.findall(r'-?\b\d+\.?\d*\b', line)]

    elif experiment_data.benchmark in'swaptions':
        execution_log =  io.TextIOWrapper(gzip.open(experiment_data.log_file, 'r'), encoding="utf-8")
        for line in execution_log:
            m = re.search(r'SwaptionPrice: (\d+\.\d+) StdError: (\d+\.\d+)', line)
            if m is not None:
                run_out.append(float(m.group(1)))
                # run_out.append(float(m.group(2)))

    elif experiment_data.benchmark in 'x264':

        ref_file = experiment_data.reference_file
        ref_images = parse_images(ref_file) # maybe this should be the original video?

        run_file = experiment_data.output_file
        run_images = parse_images(run_file)
        
        # run_out.append(os.path.getsize(run_file)) # TODO: make balanced weight.

        for ref, run in zip(ref_images, run_images):
            run_out.append(psnr(ref, run))
    
    return run_out

def calc_accuracy(output, reference):
    loss = 0

    comparison = list(zip(get_benchmark_output(output), get_benchmark_output(reference)))

    # print(len(comparison))

    for e in comparison:
        try:
            loss += abs(e[0] - e[1])/ abs(e[0])
        except:
            print("warning zero div: trying calculation ({} - {} / {})".format(e[0], e[1], e[0]))
            continue
    
    return max(1 - (loss / len(comparison)), 0)

def calc_accuracy_sum(output, reference):
    loss = 0

    output = get_benchmark_output(output)
    reference = get_benchmark_output(reference)

    loss = (abs(sum(reference)- sum(output)) / abs(sum(reference)))

    if loss > 1: loss = 1

    
    return 1 - loss

def calc_file_size_accuracy(output: ExpData, reference: ExpData):
    out_size = os.path.getsize(output.output_file)
    ref_size = os.path.getsize(reference.output_file)

    return 1- (out_size - ref_size) / out_size

def calc_speedup(output: ExpData, reference: ExpData):
    responce_time_out = 0
    responce_time_ref = 0

    for line in io.TextIOWrapper(gzip.open(output.log_file, 'r'), encoding='utf-8'):
        m = re.search(r'Task (\d+) \(Response/Service/Wait\) Time \(ns\)\s+:\s+(\d+)\s+(\d+)\s+(\d+)', line)
        if m is not None:
            responce_time_out = int(m.group(2))

    for line in io.TextIOWrapper(gzip.open(reference.log_file, 'r'), encoding='utf-8'):
        m = re.search(r'Task (\d+) \(Response/Service/Wait\) Time \(ns\)\s+:\s+(\d+)\s+(\d+)\s+(\d+)', line)
        if m is not None:
            responce_time_ref = int(m.group(2))

    return responce_time_ref / responce_time_out

def divide_chunks(l, n):   
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n]    

def make_dataframe():
    results = compile_testdata("swaptions_surface_3_a0_b1:", RESULTS_DIR, 'swaptions')
    reference = results[0]

    data = {
        # 'x': [ 0, 20, 40, 60],
        # 'y': [ 0, 20, 40, 60],
        'pr_vec': [],
        # 'speed-up': [],
        'accuracy': [],
        'sum-accuracy': [],
        'heart-rate': [],
        'relative-rate':[]
    }

    print(results[0].pr_vector)

    for r in results:
        # catch for the broken log
        if r.pr_vector == ['40','60', '0']:
            continue

        accuracy = calc_accuracy(r, reference)
        sum_accuracy = calc_accuracy_sum(r, reference) 
        speedup = calc_speedup(r, reference)

        data['pr_vec'].append([int(pr) for pr in r.pr_vector])
        data['accuracy'].append(accuracy)
        data['sum-accuracy'].append(sum_accuracy)
        data['heart-rate'].append(r.hb_df['Instant Rate'].iloc[-1])
        data['relative-rate'].append( r.hb_df['Instant Rate'].iloc[-1] - reference.hb_df['Instant Rate'].iloc[-1])

    df = pd.DataFrame(data)
    print(df)

make_dataframe()