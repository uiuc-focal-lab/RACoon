import sys
import glob
import shutil
import numpy as np
sys.path.append('./')
from src.common import Dataset
import raven.src.config as config
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

# plt.axhline(0, color='black', linewidth=.5)

class DataStruct:
    def __init__(self, eps = None,
    individual = None,
    IO = None,
    individual_refinement = None,
    individual_refinement_MILP = None,
    cross_ex = None,
    cross_ex_MILP = None,
    individual_time = None,
    IO_time = None,
    individual_refinement_time = None,
    individual_refinement_MILP_time = None,
    cross_ex_time = None,
    cross_ex_MILP_time = None):
        self.eps = eps
        self.individual = individual
        self.IO = IO
        self.individual_refinement = individual_refinement
        self.individual_refinement_MILP = individual_refinement_MILP
        self.cross_ex = cross_ex
        self.cross_ex_MILP = cross_ex_MILP
        self.individual_time = individual_time
        self.IO_time = IO_time
        self.individual_refinement_time = individual_refinement_time
        self.individual_refinement_MILP_time = individual_refinement_MILP_time
        self.cross_ex_time = cross_ex_time
        self.cross_ex_MILP_time = cross_ex_MILP_time

class DataStructList:
    def __init__(self):
        self.eps_list = []
        self.eps_res_map = {}
        self.sorted_res_list = None
    
    def add_res(self, res : DataStruct):
        if self.sorted_res_list is not None:
            raise ValueError(f'Can not add more result')
        self.eps_list.append(res.eps)
        self.eps_res_map[res.eps] = res
    
    def sort_list(self):
        self.eps_list.sort()
        self.sorted_res_list = []
        for eps in self.eps_list:
            self.sorted_res_list.append(self.eps_res_map[eps])

    def generate_plot(self, dir, filename, dataset, eps_list, individual_list, IO_list, 
                    cross_ex_MILP_list, individual_refinement_list=None, 
                    individual_refinement_MILP_list=None, cross_ex_list=None):
        sns.set_style("darkgrid")
        # Plot the three line plots
        plt.figure(figsize=(6, 4.5))  # Optional: set the figure size
        ax = plt.axes()
        
        # Setting the background color of the plot 
        # using set_facecolor() method
        ax.set_facecolor("lightgrey")
        plt.plot(eps_list, individual_list, marker='D', label='Non-relational', linestyle='-', color='cornflowerblue')
        plt.plot(eps_list, IO_list, marker='s', label='I/O Formulation', linestyle='-', color='indianred')
        plt.plot(eps_list, cross_ex_MILP_list, marker='o', label='RACoon', linestyle='-', color='darkseagreen')
        if individual_refinement_list is not None:
            plt.plot(eps_list, individual_refinement_list, marker='+', label='Individual Refinement', linestyle='-', color='black')
        if individual_refinement_MILP_list is not None:
            plt.plot(eps_list, individual_refinement_MILP_list, marker='*',
                      label='Individual Refinement MILP', linestyle='-', color='darkviolet')
        if cross_ex_list is not None:
            plt.plot(eps_list, cross_ex_list, marker='x', label='CrossEx Refinement', linestyle='-', color='cyan')


        plt.legend(loc=4, fontsize="10")
        # Add labels and a legend
        #plt.gca().yaxis.label.set(rotation='horizontal', ha='left');
        bbox = ax.get_yticklabels()[-1].get_window_extent()
        x,_ = ax.transAxes.inverted().transform([bbox.x0, bbox.y0])
        # ax.set_yticks([10, 20, 30, 40, 50, 60, 70, 80, 90], minor=True)
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.grid(axis='x', which='both')
        ax.set_ylim((0.0 if individual_list[-1] < 50 else 50), 100.0 )
        if dataset == Dataset.CIFAR10:
            plt.xlabel('Epsilon (*/255)', fontsize=15)
        else:
            plt.xlabel('Epsilon', fontsize = 15)

        ax.set_title('Average Worst Case UAP Accuracy (%)', fontsize=15, ha='left', x=x)
        plt.legend(loc=3, fontsize="12")
        plt.tight_layout(pad=0.5)
        plot_name = dir + '/' + filename
        plt.savefig(plot_name, dpi=600,  bbox_inches='tight')

    def plot(self, dir, filename, dataset, threshold=[None, None], draw_full=False):
        filtered_eps_list = []
        individual_list = []
        IO_list = [] 
        cross_ex_MILP_list = [] 
        individual_refinement_list=None if not draw_full else [] 
        individual_refinement_MILP_list=None if not draw_full else []
        cross_ex_list=None if not draw_full else []
        if self.sorted_res_list is None:
            self.sort_list()       
        for i, eps in enumerate(self.eps_list):
            assert self.sorted_res_list[i].eps == eps
            if threshold[0] is not None and threshold[0] > eps:
                continue
            if threshold[1] is not None and threshold[1] < eps:
                continue
            res = self.sorted_res_list[i]
            filtered_eps_list.append(eps)
            individual_list.append(res.individual)
            IO_list.append(res.IO)
            cross_ex_MILP_list.append(res.cross_ex_MILP)
            if not draw_full:
                continue
            individual_refinement_list.append(res.individual_refinement)
            individual_refinement_MILP_list.append(res.individual_refinement_MILP)
            cross_ex_list.append(res.cross_ex)
        
        self.generate_plot(dir, filename,  dataset, filtered_eps_list, individual_list, IO_list, 
                    cross_ex_MILP_list)
        if draw_full:
            filename = filename.split('.')[0]
            filename = f'{filename}_full.png'
            self.generate_plot(dir, filename,  dataset, filtered_eps_list, individual_list, IO_list, 
                cross_ex_MILP_list, individual_refinement_list=individual_refinement_list, 
                individual_refinement_MILP_list=individual_refinement_MILP_list, cross_ex_list=cross_ex_list)

    def process_data(self, dir, filename, dataset, process_full=False):
        if self.sorted_res_list is None:
            self.sort_list()
        filtered_eps_list = []
        individual_list = []
        IO_list = [] 
        cross_ex_MILP_list = [] 
        individual_refinement_list=None if not process_full else [] 
        individual_refinement_MILP_list=None if not process_full else []
        cross_ex_list=None if not process_full else []
        individual_time_list = []
        IO_time_list = [] 
        cross_ex_MILP_time_list = [] 
        individual_refinement_time_list=None if not process_full else [] 
        individual_refinement_MILP_time_list=None if not process_full else []
        cross_ex_time_list=None if not process_full else []
        if self.sorted_res_list is None:
            self.sort_list()       
        for i, eps in enumerate(self.eps_list):
            assert self.sorted_res_list[i].eps == eps
            res = self.sorted_res_list[i]
            filtered_eps_list.append(eps)
            individual_list.append(res.individual)
            individual_time_list.append("{:.2f}".format(res.individual_time))
            IO_list.append(res.IO)
            IO_time_list.append("{:.2f}".format(res.IO_time))
            cross_ex_MILP_list.append(res.cross_ex_MILP)
            cross_ex_MILP_time_list.append("{:.2f}".format(res.cross_ex_MILP_time))            
            if not process_full:
                continue
            individual_refinement_list.append(res.individual_refinement)
            individual_refinement_time_list.append("{:.2f}".format(res.individual_refinement_time))
            individual_refinement_MILP_list.append(res.individual_refinement_MILP)
            cross_ex_list.append(res.cross_ex)
            individual_refinement_MILP_time_list.append("{:.2f}".format(res.individual_refinement_MILP_time))
            cross_ex_time_list.append("{:.2f}".format(res.cross_ex_time))
        
        file_path = dir + '/' + filename
        with open(file_path, 'w+') as file:
            for i, eps in enumerate(filtered_eps_list):
                # print(f'eps {eps}')
                s = f'& {individual_list[i]} & {individual_time_list[i]} & {IO_list[i]} & {IO_time_list[i]} & {cross_ex_MILP_list[i]}'
                s += f'\;(' +'\\textcolor{mgreen}{'+ f'+{cross_ex_MILP_list[i] - IO_list[i]}' +'})'
                s += f'& {cross_ex_MILP_time_list[i]} ' +"\\\\"
                if dataset == Dataset.MNIST:
                    eps_str = "{:.3f}".format(eps) 
                else: 
                    eps_str = ("{:.1f}".format(eps) + "/255")
                s = eps_str + ' ' + s + '\n'
                file.write(s)
        file.close()


def copy_files(source_directory, destination_directory):
    # Use shutil.copy() to copy all files from the source directory to the destination directory
    try:
        shutil.copytree(source_directory, destination_directory)
        print("All files copied successfully.")
    except FileNotFoundError:
        print("Source directory not found.")
    except FileExistsError:
        print("Destination directory already exists.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")



def process_file(file, eps):
    individual = None
    IO = None
    individual_refinement = None
    individual_refinement_MILP = None
    cross_ex = None
    cross_ex_MILP = None
    individual_time = None
    IO_time = None
    individual_refinement_time = None
    individual_refinement_MILP_time = None
    cross_ex_time = None
    cross_ex_MILP_time = None

    for line in file:
        tokens = line.split(' ')
        if tokens[0] == 'individual':
            if tokens[1] == 'acc':
                individual =  float(tokens[2])
            if tokens[1] == 'time':
                individual_time =  float(tokens[2])
            continue
        if tokens[0] == 'baseline':
            if tokens[1] == 'acc':
                IO =  float(tokens[2])
            if tokens[1] == 'time':
                IO_time =  float(tokens[2])
            continue
        
        if tokens[0] == 'IndivRefine':
            if tokens[1] == 'acc':
                individual_refinement =  float(tokens[2])
            if tokens[1] == 'time':
                individual_refinement_time =  float(tokens[2])
            continue    
        if tokens[0] == 'IndivRefineMILP':
            if tokens[1] == 'acc':
                individual_refinement_MILP =  float(tokens[2])
            if tokens[1] == 'time':
                individual_refinement_MILP_time =  float(tokens[2])
            continue    
        if tokens[0] == 'CrossEx':
            if tokens[1] == 'acc':
                cross_ex =  float(tokens[2])
            if tokens[1] == 'time':
                cross_ex_time =  float(tokens[2])
            continue
        if tokens[0] == 'final':
            if tokens[1] == 'acc':
                cross_ex_MILP =  float(tokens[2])
            if tokens[1] == 'time':
                cross_ex_MILP_time =  float(tokens[2])
            continue

    return DataStruct(eps=eps, individual=individual, IO=IO, individual_refinement=individual_refinement,
                      individual_refinement_MILP=individual_refinement_MILP, cross_ex=cross_ex,
                      cross_ex_MILP=cross_ex_MILP, individual_time=individual_time, IO_time=IO_time,
                      individual_refinement_time=individual_refinement_time, 
                      individual_refinement_MILP_time=individual_refinement_MILP_time,
                      cross_ex_time=cross_ex_time, cross_ex_MILP_time=cross_ex_MILP_time)



def process_file_name(full_name):
    filename = full_name.split('/')[-1]
    split_name = filename.split('_')
    prop_count = int(split_name[-2])
    eps = split_name[-1].split('.da')[0]
    eps = float(eps)
    return prop_count, eps

def read_files(net_name, source_directory, thershold):
    directory = './raw_results/'  # Replace with the actual directory path
    # remove the raw results if exist
    shutil.rmtree('./raw_results')
    # Copy reults if empty.
    copy_files(source_directory=source_directory, destination_directory='./raw_results')
    # Define the pattern you want to search for
    pattern = f'{net_name}*.dat'  # For example, open all files with a .txt extension


    # Use the glob.glob() function to find files that match the pattern
    file_list = glob.glob(directory + '/' + pattern)
    # print(file_list)

    prop_count_result = {}
    # Loop through the list of matching files and open them
    for file_path in file_list:
        prop_count, eps = process_file_name(full_name=file_path)
        # print(f'prop count {prop_count} eps {eps}')
        with open(file_path, 'r') as file:
            data_struct = process_file(file=file, eps=eps)
            if prop_count not in prop_count_result.keys():
                prop_count_result[prop_count] = DataStructList()
            prop_count_result[prop_count].add_res(data_struct)
    return prop_count_result

def main():
    mnist_net_names = [config.MNIST_CONV_SMALL_DIFFAI,
                       config.MNIST_CROWN_IBP,
                        config.MNIST_FFN_01,
                        config.MNIST_CROWN_IBP_MED,
                        config.MNIST_CONV_SMALL,
                        config.MNIST_CONV_PGD,
                        config.MNIST_CONV_BIG,]
    thresholds = {}
    thresholds[config.MNIST_CONV_SMALL_DIFFAI] = [None, None]
    thresholds[config.MNIST_CROWN_IBP] = [None, None]
    thresholds[config.MNIST_FFN_01] = [None, None]
    thresholds[config.MNIST_CROWN_IBP_MED] = [None, 0.3003]
    thresholds[config.MNIST_CONV_PGD] = [None, None]
    thresholds[config.MNIST_CONV_SMALL] = [None, None]
    thresholds[config.MNIST_CONV_BIG] = [None, 0.2003]

    for net_name in mnist_net_names:
       prop_count_result =  read_files(net_name=net_name, source_directory='./icml_results_diff_k/', thershold=None)
       for count_prop, res_list in prop_count_result.items():
           res_list.sort_list()
           filename = f'{net_name}_{count_prop}.png'
           dir = './plots_diff_k'
           res_list.plot(dir=dir, filename=filename, dataset=Dataset.MNIST, threshold=thresholds[net_name])
           filename = f'{net_name}_{count_prop}.dat'
           dir = './process_res_diff_k'
           res_list.process_data(dir=dir, filename=filename, dataset=Dataset.MNIST, process_full=False)
           
    cifar_net_names = [config.CIFAR_CROWN_IBP,
                       config.CIFAR_CROWN_IBP_MEDIUM,
                        config.CIFAR_CONV_COLT,
                        config.CIFAR_CONV_DIFFAI,
                        config.CIFAR_CONV_SMALL,
                        config.CIFAR_CONV_SMALL_PGD,
                        config.CIFAR_CONV_BIG,]
    thresholds = {}
    thresholds[config.CIFAR_CROWN_IBP] = [None, None]
    thresholds[config.CIFAR_CROWN_IBP_MEDIUM] = [None, None]
    thresholds[config.CIFAR_CONV_COLT] = [None, None]
    thresholds[config.CIFAR_CONV_DIFFAI] = [None, None]
    thresholds[config.CIFAR_CONV_SMALL_PGD] = [0.5, 4.0]
    thresholds[config.CIFAR_CONV_SMALL] = [0.5, 2.0]
    thresholds[config.CIFAR_CONV_BIG] = [None, None]

    for net_name in cifar_net_names:
       prop_count_result =  read_files(net_name=net_name, source_directory='./icml_results_diff_k/', thershold=None)
       for count_prop, res_list in prop_count_result.items():
           res_list.sort_list()
           filename = f'{net_name}_{count_prop}.png'
           dir = './plots_diff_k'
           res_list.plot(dir=dir, filename=filename, dataset=Dataset.CIFAR10, threshold=thresholds[net_name])
           filename = f'{net_name}_{count_prop}.dat'
           dir = './process_res_diff_k'
           res_list.process_data(dir=dir, filename=filename, dataset=Dataset.CIFAR10, process_full=False)

    return

if __name__ == "__main__":
    main()
