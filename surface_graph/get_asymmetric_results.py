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

    loss = (abs(sum(reference)- sum(output)) / sum(reference))
    
    print(1 - loss)

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


def print_qos_statistics():
    results = compile_testdata("swaptions_surface_graph:", RESULTS_DIR, 'swaptions')

    for dirname in sorted(os.listdir(RESULTS_DIR)):
        for key in ('bodytrack', 'swaptions', 'x264', 'blackscholes'):
            if key in dirname:
                experiment_data: tuple[ExpData, str] = (compile_testdata(dirname, RESULTS_DIR, key), dirname.split(':')[1])

                if key == 'x264':
                    experiment_data[0].reference_file = results[key][1].output_file
                
                results[key][0].append(experiment_data)

    for benchmark, result in results.items():
        print(benchmark)
        reference = result[1]
        for loop_result in result[0]:
            accuracy = calc_accuracy(loop_result[0], reference)
            sum_acc = calc_accuracy_sum(loop_result[0], reference)

            if benchmark in 'x264':
                accuracy *= .5
                sum_acc *= .5

                fs_accuracy = calc_file_size_accuracy(loop_result[0], reference)*0.5
                accuracy += fs_accuracy
                sum_acc += fs_accuracy

            speedup = calc_speedup(loop_result[0], reference)
            print("loop: {}, speedup: {}, average_accuracy: {}, sum_accuracy:{}".format(loop_result[1], speedup, accuracy, sum_acc))
            
    return

def divide_chunks(l, n):   
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 

def plot_suface_graph():
    line_marker = dict(color='#0066FF', width=5)
    layout = go.Layout(
        title='Speed-up and accuracy loss of Pr_a and Pr_b in Swaptions',
        width=1600, height=800,
        scene=dict(
            xaxis=dict(
                title={'text':"Perforation Rate A [%]"},
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            yaxis=dict(
                title={'text':"Perforation Rate B [%]"},
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            zaxis=dict(
                title={'text':"Speed up of responce time against 0% Perforation"},
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            )
        ),
        showlegend=False,
    )
    
    results = compile_testdata("swaptions_surface_3_a0_b1:", RESULTS_DIR, 'swaptions')
    reference = results[0]

    data = {
        'x': [ 0, 20, 40, 60],
        'y': [ 0, 20, 40, 60],
        'pr_a': [],
        'pr_b': [],
        'speed-up': [],
        'accuracy': [],
    }

    for r in results:
        # catch for the broken log
        if r.pr_vector == ['40','60', '0']:
            continue

        accuracy = calc_accuracy(r, reference)
        speedup = calc_speedup(r, reference)

        data['pr_a'].append(int(r.pr_vector[0]))
        data['pr_b'].append(int(r.pr_vector[1]))
        data['speed-up'].append(speedup)
        data['accuracy'].append(accuracy)
    
    # add in missing point. 
    data['accuracy'].insert(11,0)
    data['speed-up'].insert(11,1.56)
    data['pr_a'].insert(11,40)
    data['pr_b'].insert(11,60)

    pr_a = list(divide_chunks(data['pr_a'],4))
    pr_b = list(divide_chunks(data['pr_b'],4))
    sped = list(divide_chunks(data['speed-up'],4))
    accu = list(divide_chunks(data['accuracy'],4))

    # transpose for the wireframe lines.
    pr_a_t = np.array(pr_a).T.tolist()
    pr_b_t = np.array(pr_b).T.tolist()
    sped_t = np.array(sped).T.tolist()
    accu_t = np.array(accu).T.tolist()

    fig = make_subplots(rows=1, cols=2,
                        specs=[[{'type': 'scatter3d'}, {'type': 'contour'}]],
                        subplot_titles=('Speed up graph for Pr A and Pr B', 'Accuracy contour for Pr A and Pr B',))
    
    fig.update_layout(layout)

    # build the wireframe
    for i, j, k in zip(pr_a, pr_b, sped):
        fig.append_trace(go.Scatter3d(x=i, y=j, z=k, mode='lines', line=line_marker), row=1,col=1)

    for i, j, k in zip(pr_a_t, pr_b_t, sped_t):
        fig.append_trace(go.Scatter3d(x=i, y=j, z=k, mode='lines', line=line_marker),row=1,col=1)
    
    # generate the contour
    fig.append_trace(go.Contour(x=data['x'], y= data['y'], z=accu, colorbar=dict(
        title='Average Accuracy',
        titleside='right',
    )), row=1,col=2)

    fig.update_layout(layout, scene_camera=dict(eye=dict(x=-1.5, y=-1.5, z=.1)))
    fig.update_xaxes(title_text="Perforation Rate A [%]",row=1, col=2)
    fig.update_yaxes(title_text="Perforation Rate B [%]", row=1,col=2)

    fig.write_image(os.path.join(OUTPUT_DIR,"speedup and accuracy.pdf"))
    # fig.show()
    

def plot_contour():
    results = compile_testdata("swaptions_surface_3_a0_b1:", RESULTS_DIR, 'swaptions')
    reference = results[0]

    data = {
        'x': [ 0, 20, 40, 60],
        'y': [ 0, 20, 40, 60],
        'pr_a': [],
        'pr_b': [],
        'speed-up': [],
        'accuracy': [],
    }

    for r in results:
        # catch for the broken log
        if r.pr_vector == ['40','60', '0']:
            continue

        accuracy = calc_accuracy_sum(r, reference)
        speedup = calc_speedup(r, reference)

        data['pr_a'].append(int(r.pr_vector[0]))
        data['pr_b'].append(int(r.pr_vector[1]))
        data['speed-up'].append(speedup)
        data['accuracy'].append(accuracy)
    
    # add in missing point. 
    data['accuracy'].insert(11,0)
    data['speed-up'].insert(11,1.56)
    data['pr_a'].insert(11,40)
    data['pr_b'].insert(11,60)

    accu = list(divide_chunks(data['accuracy'],4))
    
    contour = go.Figure(go.Contour(x=data['x'], y= data['y'], z=accu, colorbar=dict(
        title='Sum Accuracy',
        titleside='right',
    )))

    contour.update_layout(go.Layout(
        title='Accuracy Contour of Pr A and Pr B in Swaptions',
        width=400, height=400))
    contour.update_xaxes(title_text="Perforation Rate A [%]")
    contour.update_yaxes(title_text="Perforation Rate B [%]")
    contour.write_image(os.path.join(OUTPUT_DIR,"accuracy-contour.pdf"))

def plot_speedup():
    line_marker = dict(color='#0066FF', width=5)
    layout = go.Layout(
        width=800, height=700,
        scene=dict(
            xaxis=dict(
                title={'text':"Perforation Rate A [%]"},
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            yaxis=dict(
                title={'text':"Perforation Rate B [%]"},
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            zaxis=dict(
                title={'text':"Speed up of responce time against 0% Perforation"},
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            )
        ),
        showlegend=False,
    )

    results = compile_testdata("swaptions_surface_3_a0_b1:", RESULTS_DIR, 'swaptions')
    reference = results[0]

    data = {
        'x': [ 0, 20, 40, 60],
        'y': [ 0, 20, 40, 60],
        'pr_a': [],
        'pr_b': [],
        'speed-up': [],
        'accuracy': [],
    }

    for r in results:
        # catch for the broken log
        if r.pr_vector == ['40','60', '0']:
            continue

        accuracy = calc_accuracy(r, reference)
        speedup = calc_speedup(r, reference)

        data['pr_a'].append(int(r.pr_vector[0]))
        data['pr_b'].append(int(r.pr_vector[1]))
        data['speed-up'].append(speedup)
        data['accuracy'].append(accuracy)
    
    # add in missing point. 
    data['accuracy'].insert(11,0)
    data['speed-up'].insert(11,1.56)
    data['pr_a'].insert(11,40)
    data['pr_b'].insert(11,60)

    pr_a = list(divide_chunks(data['pr_a'],4))
    pr_b = list(divide_chunks(data['pr_b'],4))
    sped = list(divide_chunks(data['speed-up'],4))
    accu = list(divide_chunks(data['accuracy'],4))

    # transpose for the wireframe lines.
    pr_a_t = np.array(pr_a).T.tolist()
    pr_b_t = np.array(pr_b).T.tolist()
    sped_t = np.array(sped).T.tolist()
    accu_t = np.array(accu).T.tolist()

    lines = []


    # build the wireframe
    for i, j, k in zip(pr_a, pr_b, sped):
        lines.append(go.Scatter3d(x=i, y=j, z=k, mode='lines', line=line_marker))

    for i, j, k in zip(pr_a_t, pr_b_t, sped_t):
        lines.append(go.Scatter3d(x=i, y=j, z=k, mode='lines', line=line_marker))
    
    fig = go.Figure(data=lines, layout=layout)
    fig.update_layout(scene_camera=dict(eye=dict(x=-1.5, y=-1.5, z=.1)))

    fig.write_image(os.path.join(OUTPUT_DIR, "speed-up wireframe.pdf"))

plot_suface_graph()
plot_contour()
plot_speedup()