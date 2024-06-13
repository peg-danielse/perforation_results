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


RESULTS_DIR = os.path.join(os.getenv('RESULTS_ROOT'), 'motivation', 'results')
OUTPUT_DIR = os.path.join(os.getenv("RESULTS_ROOT"), 'motivation')

class ExpData:
    def __init__(self):
        self.benchmark: str = ''
        self.output_file: str = ''
        self.log_file: str = ''
        self.heat_file: str =  ''
        self.hb_df = pd.DataFrame()


def compile_testdata(label: str) -> ExpData:
    data_path =  ''
    
    for dirname in sorted(os.listdir(RESULTS_DIR)):
        if label in dirname:
            data_path= os.path.join(RESULTS_DIR, dirname)
            break

    if data_path == '': 
        print("Warning: result not found.")
        return None

    data: ExpData = ExpData()           
        
    # save reference to output file and log_file
    for file in os.listdir(data_path):
        if 'output.txt' in file:
            data.output_file = os.path.join(data_path, file)
        
        if 'poses.txt' in file:
            data.output_file = os.path.join(data_path, file)

        if 'execution.log.gz' in file:
            data.log_file = os.path.join(data_path, file)

        if 'PeriodicThermal.log.gz' in file:
            data.heat_file = os.path.join(data_path, file)
            
        if 'hb.log' in file:
            file_df = pd.read_csv(os.path.join(data_path, file), sep='\t')

            data.hb_df = pd.concat([data.hb_df, file_df])

    return data


def parse_images(video_path: str) -> list:
    encoded_vid = cv2.VideoCapture(video_path)
            
    frames = []
    succ, img = encoded_vid.read()
    while succ:
        frames.append(img) 
        succ, img = encoded_vid.read()

    return frames


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


#TODO: bodytrack output getto.
def get_bodytrack_output(data : ExpData):
    file =  open(data.output_file, 'r')
    
    run_out = []
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

# TODO: get frequency from the file.
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


def set_style():
    plt.rcParams['font.family'] = 'Arial'
    custom_params = {"axes.spines.right": False, 
                    "axes.spines.top": False,
                    "axes.spines.left": True,
                    "axes.spines.bottom": True,
                    #create dashes next to ticks
                    "xtick.bottom": True,
                    "ytick.left": True,
                    
                    "axes.edgecolor": "black",

                    "axes.grid": True,
                    "axes.linewidth": 1.5, 
                    "axes.facecolor": "white", 
                    "grid.color": "lightgray",
                    
                    }

    sns.set_theme(style="whitegrid", rc=custom_params)
    sns.set_palette("deep")


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


set_style()

plot_swaptions_perforation_results()
# plot_swaptions_motivational_example()
# plot_bodytrack_perforation_results()
# plot_bodytrack_motivational_example()
