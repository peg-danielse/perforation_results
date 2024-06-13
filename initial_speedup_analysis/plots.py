import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def qos_plot_canneal():
    raw_output = [9.27605e+07, 9.35351e+07, 9.47473e+07, 9.56478e+07]

    qos = [ 1 - abs((e - raw_output[0]) / e) for e in raw_output]

    sns.barplot(qos)
    plt.xticks([0,1,2,3], ['baseline', '20%', '50%', '70%'])

    plt.savefig('./qos_loss.pdf')
    plt.clf()

def speedup_canneal():
    runtime = [619524499, 583418627, 560957543, 558351785]

    speedup = [ runtime[0]/e for e in runtime]

    sns.lineplot(y=speedup, x=[0,1,2,3])
    plt.xticks([0,1,2,3], ['baseline', '20%', '50%', '70%'])

    plt.savefig('./speedup.pdf')
    plt.clf()

def speedup_canneal_par_sec():
    runtime = [ 473602400 - 439019700, 481615300 - 444137800, 458959400 - 433989700, 456323600 - 440134600]

    speedup = [ runtime[0]/e for e in runtime]

    sns.lineplot(y=speedup, x=[0,1,2,3])
    plt.xticks([0,1,2,3], ['baseline', '20%', '50%', '70%'])

    plt.savefig('./speedup_par_sec.pdf')
    plt.clf()

qos_plot_canneal()
speedup_canneal()
speedup_canneal_par_sec()
