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
    def __init__(self, iter = None,
    individual = None,
    cross_ex = None):
        self.iter = iter
        self.individual = individual
        self.cross_ex = cross_ex

class DataStructList:
    def __init__(self):
        self.iter_list = []
        self.iter_res_map = {}
        self.sorted_res_list = None
    
    def add_res(self, res : DataStruct):
        if self.sorted_res_list is not None:
            raise ValueError(f'Can not add more result')
        self.iter_list.append(res.iter)
        self.iter_res_map[res.iter] = res
    
    def sort_list(self):
        self.iter_list.sort()
        self.sorted_res_list = []
        for iter in self.iter_list:
            self.sorted_res_list.append(self.iter_res_map[iter])

    def generate_plot(self, dir, filename, iter_list, individual_list, cross_ex_list):
        sns.set_style("darkgrid")
        # Plot the three line plots
        plt.figure(figsize=(6, 4.5))  # Optional: set the figure size
        ax = plt.axes()
        plt.axhline(0, color='white')

        # Setting the background color of the plot 
        # using set_facecolor() method
        ax.set_facecolor("lightgrey")
        plt.plot(iter_list, individual_list, marker='.', label=r'Individual Refinement $t_{i}(\overline{G})$', linestyle='-', color='cornflowerblue')
        plt.plot(iter_list, cross_ex_list, marker='+', label=r'Cross Execution Refinement $t_{i}(G)$', linestyle='-', color='indianred')
    

        plt.legend(loc=2, fontsize="10")
        # Add labels and a legend
        #plt.gca().yaxis.label.set(rotation='horizontal', ha='left');
        bbox = ax.get_yticklabels()[-1].get_window_extent()
        x,_ = ax.transAxes.inverted().transform([bbox.x0, bbox.y0])
        # ax.set_yticks([10, 20, 30, 40, 50, 60, 70, 80, 90], minor=True)
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.grid(axis='x', which='both')
        # ax.set_ylim((0.0 if individual_list[-1] < 50 else 50), 100.0 )

        plt.xlabel( r'Iteration Count $(i)$', fontsize = 15)

        ax.set_title(r'Iterationwise Lower bound ($t_i$)', fontsize=15, ha='left', x=x)
        plt.legend(loc=4, fontsize="12")
        plt.tight_layout(pad=0.5)
        plot_name = dir + '/' + filename
        plt.savefig(plot_name, dpi=600,  bbox_inches='tight')

    def plot(self, dir, filename):
        iter_list = []
        individual_list = []
        cross_ex_list= []
        if self.sorted_res_list is None:
            self.sort_list()       
        for i, iter in enumerate(self.iter_list):
            assert self.sorted_res_list[i].iter == iter
            res = self.sorted_res_list[i]
            iter_list.append(iter)
            individual_list.append(res.individual)
            cross_ex_list.append(res.cross_ex)        
        self.generate_plot(dir, filename,  iter_list, individual_list, cross_ex_list)


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



def process_file(file):
    data_list = DataStructList()
    count = 0
    for line in file:
        tokens = line.split(' ')
        if tokens[0] == 'bounds:':
            individual = float(tokens[1]) if count <= 0 else max(individual, float(tokens[1]))
            cross_ex = float(tokens[2]) if count <= 0 else max(cross_ex, float(tokens[2]))
            res= DataStruct(iter=count, individual=individual, cross_ex=cross_ex)
            data_list.add_res(res=res)
        count +=1
    return data_list



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
        prop_count, _ = process_file_name(full_name=file_path)
        # print(f'prop count {prop_count} eps {eps}')
        with open(file_path, 'r') as file:
            data_struct = process_file(file=file)
            prop_count_result[prop_count] = data_struct
    return prop_count_result

def main():
    mnist_net_names = [config.MNIST_CONV_SMALL_DIFFAI,
                        config.MNIST_CONV_PGD, 
                        config.CIFAR_CONV_DIFFAI,
                        config.CIFAR_CONV_SMALL_PGD,]

    for net_name in mnist_net_names:
       prop_count_result =  read_files(net_name=net_name, source_directory='./bounds_ablation/', thershold=None)
       for count_prop, res_list in prop_count_result.items():
           res_list.sort_list()
           filename = f'{net_name}_{count_prop}.png'
           dir = './plot_bound'
           res_list.plot(dir=dir, filename=filename)
           
    # cifar_net_names = [config.CIFAR_CONV_DIFFAI,
    #                     config.CIFAR_CONV_SMALL_PGD,]
    # thresholds = {}
    # thresholds[config.CIFAR_CROWN_IBP] = [None, None]
    # thresholds[config.CIFAR_CROWN_IBP_MEDIUM] = [None, None]
    # thresholds[config.CIFAR_CONV_COLT] = [None, None]
    # thresholds[config.CIFAR_CONV_DIFFAI] = [None, None]
    # thresholds[config.CIFAR_CONV_SMALL_PGD] = [2.0, 4.0]
    # thresholds[config.CIFAR_CONV_SMALL] = [1.0, 2.0]
    # thresholds[config.CIFAR_CONV_BIG] = [None, None]

    # for net_name in cifar_net_names:
    #    prop_count_result =  read_files(net_name=net_name, source_directory='./icml_results_new/', thershold=None)
    #    for count_prop, res_list in prop_count_result.items():
    #        res_list.sort_list()
    #        filename = f'{net_name}_{count_prop}.png'
    #        dir = './plots'
    #        res_list.plot(dir=dir, filename=filename, dataset=Dataset.CIFAR10, threshold=thresholds[net_name])

    return

if __name__ == "__main__":
    main()
