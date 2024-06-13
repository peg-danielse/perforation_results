import os.path, io, sys, gzip
import re, cv2
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

from matplotlib.axes import Axes

from pprint import pprint


RESULTS_DIR = os.path.join(os.getenv('RESULTS_ROOT'), 'asymmetric_perforation_profile', 'results')
REFERENCE_DIR = os.path.join(os.getenv("RESULTS_ROOT"), 'asymmetric_perforation_profile', 'reference')
OUTPUT_DIR = os.path.join(os.getenv("RESULTS_ROOT"), 'asymmetric_perforation_profile')

class ExpData:
    def __init__(self):
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


def compile_testdata(label: str, path, benchmark:str = None) -> ExpData:
    data_path =  ''

    for dirname in sorted(os.listdir(path)):
        if label in dirname:
            data_path= os.path.join(path, dirname)
            break

    if data_path == '': 
        print("Warning: result not found.")
        return None

    data: ExpData = ExpData()           
        
    # save reference to output file and log_file
    for file in sorted(os.listdir(data_path)):
        if 'appmapping.txt' in file:
            map_file = os.path.join(data_path, file)
            data.benchmark = app_mapping(map_file)

        if 'output.txt' in file:
            data.output_file = os.path.join(data_path, file)
        
        if 'poses.txt' in file:
            data.output_file = os.path.join(data_path, file)
        
        if '.264' in file: 
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
        
    return data


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


def perforation_qos_loss_plot(benchmark: str, data: ExpData, ax: Axes):
    perforation_noise = []

    for _ in data.idx:
        perforation_noise.append(0.0)
    
    output_comparison = []
        
    if benchmark in {'blackscholes', 'bodytrack'}: # text files with the outputs on lines
        for file in data.output_files:
            run_out = []
            f = open(file)
            for line in f.readlines():
                run_out += [float(num) for num in  re.findall(r'-?\b\d+\.?\d*\b', line)]
            
            output_comparison.append(run_out)
    
    elif benchmark in {'canneal'}: # one number in the execution log.
        for file in data.output_files: 
            execution_log =  io.TextIOWrapper(gzip.open(file, 'r'), encoding="utf-8")
            
            for line in execution_log:
                m = re.search(r'Final routing is: (\d+\.\d+)', line)
                if m is not None:
                    output_comparison.append([float(m.group(1))])
                    break
    
    elif benchmark in {'swaptions'}: # one number in the execution log.
        for file in data.output_files: 
            execution_log =  io.TextIOWrapper(gzip.open(file, 'r'), encoding="utf-8")
            
            run_out = []
            for line in execution_log:
                m = re.search(r'SwaptionPrice: (\d+\.\d+) StdError: (\d+\.\d+)', line)
                if m is not None:
                    run_out.append(float(m.group(1)))
                    run_out.append(float(m.group(2)))
                
            output_comparison.append(run_out)

    elif benchmark in {'x264'}: # decompose video into frames and then calculate the noise.
        ref_file = data.output_files[0]
        ref_images = parse_images(ref_file) # maybe this should be the original video?
        
        for run_file in data.output_files: 
            run_out = []
            run_images = parse_images(run_file)
            
            run_out.append(os.path.getsize(run_file))

            for ref, run in zip(ref_images, run_images):
                run_out.append(psnr(ref, run))

            print(run_out)
            output_comparison.append(run_out)

    else:
        ax.set_title(name + " QoS on simsmall")
        ax.set_ylabel("% noise")
        ax.set_xlabel("perforation rate")
        return
    
    reference = output_comparison[0]
    for index, output in enumerate(output_comparison):
        perforation_noise[index] = 100 * (abs(sum(reference)- sum(output)) / abs(sum(reference)))


def get_output(experiment_data : ExpData):
     
    execution_log =  io.TextIOWrapper(gzip.open(experiment_data.log_file, 'r'), encoding="utf-8")
    
    run_out = []
    for line in execution_log:
        m = re.search(r'SwaptionPrice: (\d+\.\d+) StdError: (\d+\.\d+)', line)
        if m is not None:
            run_out.append(float(m.group(1)))
            # run_out.append(float(m.group(2)))
                    
    return run_out


def get_bodytrack_output(data : ExpData):
    
    run_out = []
    file =  open(data.output_file, 'r')
    for line in file.readlines():
        run_out += [float(num) for num in  re.findall(r'-?\b\d+\.?\d*\b', line)]
            
    return run_out


def get_peak_temperature_traces(data: ExpData):
    heat_log =  io.TextIOWrapper(gzip.open(data.heat_file, 'r'), encoding="utf-8")
    traces = []

    for line in heat_log:
        try:
            vs = [1 * float(v) for v in line.split()]
            traces.append(vs)
        except:
            continue

    traces = list(zip(*traces))

    peak = []
    for values in zip(*traces):
        peak.append(max(values))
    return peak


def heat_speedup_accuracy_plot_swaptions(data: ExpData, pr: tuple, ax: Axes, baseline_output, frequency: int, title: str):
    # Add perforation rate and accuracy metric. 
    output_comp = list(zip(baseline_output, get_output(data)))
    
    loss = 0
    for e in output_comp:
        loss += abs(e[1] - e[0])/ abs(e[1])

    noise = (loss / len(output_comp))

    ax.text(x=75, y=90, 
            s="Frequency={}\nPr={}\nAccuracy={:.4f}\n".format(frequency, pr, noise), 
            color='black', verticalalignment='bottom')
    
    # Plot the peak temperature of the CPU.
    peaks = get_peak_temperature_traces(data)
    sns.lineplot(y=peaks, x=[i for i, v in enumerate(peaks)], ax=ax)
    ax.set_xlim(-1, 110)
    ax.set_ylim(50, 120)

    # temp constraint line
    ax.axhline(y=85, color='red', linestyle='--')
    ax.text(x=-5, y=85, s=r'$\Delta$', fontsize=10, color='red', verticalalignment='center')

    # Deadline
    ax.axvline(x=70, color='black', linestyle='dotted')
    ax.text(x=72, y=52, s="Deadline", color='grey', verticalalignment='bottom')
    ax.fill_betweenx(y=np.linspace(0, 120, 400), x1=70, x2=110, color='grey', alpha=0.1)

    ax.set_title(title)
    ax.set_xlabel("Time [S]")
    ax.set_ylabel("Temperature [°C]")
    return


def heat_speedup_accuracy_plot_any(data: ExpData, pr: tuple, ax: Axes, baseline_output, frequency: int, title: str):
    # Add perforation rate and accuracy metric. 
    output_comp = list(zip(baseline_output, get_output(data)))
    
    loss = 0
    for e in output_comp:
        loss += abs(e[1] - e[0])/ abs(e[1])

    noise = 100* (loss / len(output_comp))

    ax.text(x=20, y=140, 
            s="Frequency={}\nPr={}\nAccuracy={:.4f}\n".format(frequency, pr, noise), 
            color='black', verticalalignment='bottom')
    
    # Plot the peak temperature of the CPU.
    peaks = get_peak_temperature_traces(data)
    sns.lineplot(y=peaks, x=[i for i, v in enumerate(peaks)], ax=ax)
    # ax.set_xlim(-1, 110)
    # ax.set_ylim(50, 120)

    # temp constraint line
    # ax.axhline(y=85, color='red', linestyle='--')
    # ax.text(x=-5, y=85, s=r'$\Delta$', fontsize=10, color='red', verticalalignment='center')

    # Deadline
    # ax.axvline(x=70, color='black', linestyle='dotted')
    # ax.text(x=72, y=52, s="Deadline", color='grey', verticalalignment='bottom')
    # ax.fill_betweenx(y=np.linspace(0, 120, 400), x1=70, x2=110, color='grey', alpha=0.1)

    ax.set_title(title)
    ax.set_xlabel("Time [S]")
    ax.set_ylabel("Temperature [°C]")
    return


def heat_speedup_accuracy_plot_bodytrack(data: ExpData, pr: tuple, ax: Axes, baseline_output, frequency: int, title: str):
    # Add perforation rate and accuracy metric.
    output_comp = list(zip(baseline_output, get_bodytrack_output(data)))
    
    loss = 0
    for e in output_comp:
        loss += abs(e[1] - e[0])/ abs(e[1])

    accuracy = 1 -(loss / len(output_comp))

    ax.text(x=52, y=152, 
            s="Frequency={}\nPr={}\nAccuracy={:.4f}\n".format(frequency, pr, accuracy), 
            color='black', verticalalignment='bottom')

    # Plot the peak temperature of the CPU.
    peaks = get_peak_temperature_traces(data)
    sns.lineplot(y=peaks, x=[i for i, v in enumerate(peaks)], ax=ax)
    ax.set_xlim(-1, 75)
    ax.set_ylim(50, 220)

    # temp constraint line
    ax.axhline(y=150, color='red', linestyle='--')
    ax.text(x=-5, y=150, s=r'$\Delta$', fontsize=10, color='red', verticalalignment='center')

    # Deadline
    ax.axvline(x=50, color='black', linestyle='dotted')
    ax.text(x=52, y=52, s="Deadline", color='grey', verticalalignment='bottom')
    ax.fill_betweenx(y=np.linspace(0, 220, 400), x1=50, x2=75, color='grey', alpha=0.1)

    ax.set_title(title)
    ax.set_xlabel("Time [S]")
    ax.set_ylabel("Temperature [°C]")
    return


def plot_swaptions_perforation_results():
    figure_data = []
    for i, folder in enumerate(sorted(os.listdir(RESULTS_DIR))):
        if "swap_profile" in folder:
            result = (compile_testdata(folder), folder.split(':')[1].split(','), 2, "result_{}".format(i))
            figure_data.append(result)


    baseline = compile_testdata("swap_profile_pr:40,0")
    baseline_output = get_output(baseline)

    fig, axs = plt.subplots(len(figure_data), 1, figsize=(6, 2.5*len(figure_data)))
    flat_ax = (axs.flatten())

    for i, ax in enumerate(flat_ax):
        heat_speedup_accuracy_plot_any(figure_data[i][0], figure_data[i][1], ax, baseline_output, figure_data[i][2], figure_data[i][3])
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "perforation-results_swaptions_{}.pdf".format(datetime.today())))


def plot_swaptions_motivational_example():
    # Load traces.
    three_ghz = (compile_testdata("results_2024-05-13_13.25_3.0GHz+maxFreq_parsec-swaptions-simsmall-3_exp_temp_pr:0"), (0), 3, "at 3Ghz getting too hot")
    two_ghz = (compile_testdata("exp_qos_pr:0,0"), (0), 2, "at 2Ghz not meeting deadline")
    two_perf_2 = (compile_testdata("exp_qos_pr:35,35"),(35), 2, "at 2Ghz with symetric perforation")
    two_asym_1 = (compile_testdata("exp_qos_pr:35,20"), (35, 20), 2, "at 2Ghz with asymetric perforation")

    figure_data = [three_ghz, two_ghz, two_perf_2, two_asym_1]


    # Make figure
    baseline_output = get_output(two_ghz[0])

    fig, axs = plt.subplots(4, 1, figsize=(6, 10))
    flat_ax = (axs.flatten())

    for i, ax in enumerate(flat_ax):
        heat_speedup_accuracy_plot(figure_data[i][0], figure_data[i][1], ax, baseline_output, figure_data[i][2], figure_data[i][3])
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "motivational-example_swaptions_{}.pdf".format(datetime.today())))


def plot_bodytrack_perforation_results():
    figure_data = []
    for i, folder in enumerate(sorted(os.listdir(RESULTS_DIR))):
        if "8_exp_bdytrk_pr" in folder:
            result = (compile_testdata(folder), folder.split(':')[1].split(','), 4, folder)
            figure_data.append(result)


    baseline = compile_testdata("8_exp_bdytrk_pr:0,0,0") #(compile_testdata("exp_qos_pr:0,0"), (0), 2, "at 2Ghz not meeting deadline")
    baseline_output = get_bodytrack_output(baseline)

    fig, axs = plt.subplots(len(figure_data), 1, figsize=(10, 2.5*len(figure_data)))
    flat_ax = (axs.flatten())

    for i, ax in enumerate(flat_ax):
        heat_speedup_accuracy_plot_bodytrack(figure_data[i][0], figure_data[i][1], ax, baseline_output, figure_data[i][2], figure_data[i][3])
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "perforation-results_bodytrack_{}.pdf".format(datetime.today())))


def plot_bodytrack_motivational_example():

    four  = (compile_testdata('results_2024-05-22_08.26_4.0GHz+maxFreq_parsec-bodytrack-simsmall-8_exp_bdytrk_pr:0,0,0'), (0), 4, "Peak temperature at 4Ghz")
    three = (compile_testdata('results_2024-05-22_09.22_3.0GHz+maxFreq_parsec-bodytrack-simsmall-8_exp_bdytrk_pr:0,0,0'), (0), 3, "Peak temperature at 3Ghz")
    sym   = (compile_testdata('results_2024-05-21_21.02_3.0GHz+maxFreq_parsec-bodytrack-simsmall-8_exp_bdytrk_pr:70,70,60'), (70), 3, "perforation rate at 70 (change)")
    asym  = (compile_testdata('results_2024-05-21_21.02_3.0GHz+maxFreq_parsec-bodytrack-simsmall-8_exp_bdytrk_pr:70,70,60'), (70,20,80), 3, "asymetric perforation (change)")

    figure_data = [four, three, sym, asym]
    baseline = compile_testdata("results_2024-05-22_08.26_4.0GHz+maxFreq_parsec-bodytrack-simsmall-8_exp_bdytrk_pr:0,0,0") #(compile_testdata("exp_qos_pr:0,0"), (0), 2, "at 2Ghz not meeting deadline")
    baseline_output = get_bodytrack_output(baseline)

    fig, axs = plt.subplots(len(figure_data), 1, figsize=(6, 2.5*len(figure_data)))
    flat_ax = (axs.flatten())

    for i, ax in enumerate(flat_ax):
        heat_speedup_accuracy_plot_bodytrack(figure_data[i][0], figure_data[i][1], ax, baseline_output, figure_data[i][2], figure_data[i][3])
        
    plt.tight_layout()
    plt.title("Effectiveness of asymetric perforation on Bodytrack")
    plt.savefig(os.path.join(OUTPUT_DIR, "motivational-example_bodytrack_{}.pdf".format(datetime.today())))


def calc_accuracy(output, reference):
    loss = 0

    comparison = list(zip(get_benchmark_output(output), get_benchmark_output(reference)))

    for e in comparison:
        try:
            loss += abs(e[0] - e[1])/ abs(e[0])
        except:
            # print("warning zero div: trying calculation ({} - {} / {})".format(e[0], e[1], e[0]))
            continue
    
    return 1 - (loss / len(comparison))

def calc_accuracy_sum(output, reference):
    loss = 0

    output = get_benchmark_output(output)
    reference = get_benchmark_output(reference)

    loss = (abs(sum(reference)- sum(output)) / abs(sum(reference)))
    
    return 1- loss

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
    results = {"bodytrack":     ([], compile_testdata("results_2024-03-05_11.27_4.0GHz+maxFreq+slowDVFS_parsec-bodytrack-simsmall-2_pr_0_range_medium", REFERENCE_DIR, "bodytrack")),
               "swaptions":     ([], compile_testdata("results_2024-03-03_12.20_4.0GHz+maxFreq+slowDVFS_parsec-swaptions-simsmall-3_pr_0_range_medium", REFERENCE_DIR, "swaptions")),
               "x264":          ([], compile_testdata("results_2024-03-02_16.25_4.0GHz+maxFreq+slowDVFS_parsec-x264-simsmall-3_pr_0_range_medium", REFERENCE_DIR, "x264")),
               "blackscholes":  ([], compile_testdata("results_2024-03-05_14.43_4.0GHz+maxFreq+slowDVFS_parsec-blackscholes-simsmall-3_pr_0_range_medium", REFERENCE_DIR, "blackscholes"))}

    results['x264'][1].reference_file = results['x264'][1].output_file

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
            
    # 1. for each benchmark in the set bench(AoI) print the QoS and the Speed-up
    # 2. and the loop id
    return


print_qos_statistics()
