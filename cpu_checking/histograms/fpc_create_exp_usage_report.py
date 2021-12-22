#!/usr/bin/env python3

import argparse
import collections
import glob
import json
import os
import sys
import shutil
import matplotlib.pyplot as plt

# -------------------------------------------------------- #
# Globals
# -------------------------------------------------------- #

FP32_EXPONENT_SIZE = 15
FP64_EXPONENT_SIZE = 100
# Contains accumulated data (Useful for openMP and MPI programs wherein multiple traces can be generated)
accumulated_data = []
# These sets track traces recorded in accumulated data. Useful to identify existing/new traces
input_set = set()
file_set = set()
line_set = set()
# -------------------------------------------------------- #
# PATHS
# -------------------------------------------------------- #

ROOT_REPORT_NAME = 'index.html'
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_REPORT_TEMPLATE_DIR = THIS_DIR+'/../cpu_checking/histograms/report_templates'
ROOT_REPORT_TEMPLATE = ROOT_REPORT_TEMPLATE_DIR+'/'+ROOT_REPORT_NAME

# Loads a single json trace file (histogram trace)
# Returns a dictionary
def loadReport(file_name):
    f = open(file_name, 'r')
    data = json.load(f)
    f.close()
    return data


# Loads all the json trace files (histogram traces) and merges them
# Returns nothing
def loadTraces(files):
    for f in files:
        data = loadReport(f)
        for i in range(len(data)):
            if data[i]['input'] not in input_set or data[i]['file'] not in file_set or data[i]['line'] not in line_set:
                accumulated_data.append(data[i])
                input_set.add(data[i]['input'])
                file_set.add(data[i]['file'])
                line_set.add(data[i]['line'])
            else:
                for j in range(len(accumulated_data)):
                    if data[i]['input'] == accumulated_data[j]['input'] and data[i]['file'] == accumulated_data[j]['file'] and data[i]['line'] == accumulated_data[j]['line']:
                        for (exp, count) in data[i]['fp32'].items():
                            if exp in accumulated_data[j]['fp32']:
                                accumulated_data[j]['fp32'][exp] = accumulated_data[j]['fp32'][exp] + count
                            else:
                                accumulated_data[j]['fp32'][exp] = count
                        for (exp, count) in data[i]['fp64'].items():
                            if exp in accumulated_data[j]['fp64']:
                                accumulated_data[j]['fp64'][exp] = accumulated_data[j]['fp64'][exp] + count
                            else:
                                accumulated_data[j]['fp64'][exp] = count


# Create a new directory if does not exist
# Returns: nothing
def createDirectory(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)


# Accumulates quantity in data dictionary over specified key and adds to init_dict
# init_dict and data need to have same structure
# Returns: dictionary
def accumulateOverKey(init_dict, data, key):
    for exponent in data[key].keys():
        if exponent in init_dict[key]:
            init_dict[key][exponent] += data[key][exponent]
        else:
            init_dict[key][exponent] = data[key][exponent]

    return init_dict


# Merging keys from selected fields of a dictionary.
# Returns: set
def mergeKeys(data_dictionary, fields_to_merge_from):
    keys_set = set()
    for field in fields_to_merge_from:
        keys_set.update(data_dictionary[field].keys())

    return keys_set


# Initializing missing values from keys_set in field to 0
# Returns: dictionary
def cleanUpField(data_dictionary, field, keys_set):
    cleaned_up_field = dict.fromkeys(keys_set, 0)
    cleaned_up_field.update(data_dictionary[field])
    return cleaned_up_field


# Multibar plotting of exponent histogram
# Returns: nothing
def plotExponentHistogram(x_axis_values, y_axis_fp32_values, y_axis_fp64_values, destination_directory, plot_name):
    x_axis_label_position = list(range(len(x_axis_values)))
    plt.clf()
    plt.xticks(x_axis_label_position, x_axis_values)

    x_axis_label_position[:] = [number - 0.2 for number in x_axis_label_position]
    plt.bar(x_axis_label_position, y_axis_fp32_values, 0.4, label="fp32")

    x_axis_label_position[:] = [number + 0.4 for number in x_axis_label_position]
    plt.bar(x_axis_label_position, y_axis_fp64_values, 0.4, label="fp64")

    plt.legend()
    plt.xlabel('Exponent')
    plt.ylabel('Counts')

    plt.savefig(destination_directory + '/' + plot_name)

# Generator, which yields successive n-sized chunks from lst.
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Multibar plotting of exponent histogram with ranges
# Returns: nothing
def plotExponentHistogramRanges(x_axis_values, y_axis_fp32_values, y_axis_fp64_values, destination_directory, plot_name):
    # Get counts for each range in FP32
    x_fp32 = []
    y_fp32 = []
    for r in chunks(list(range(-127, 130)), FP32_EXPONENT_SIZE):
        count = 0
        for i in r:
            for j in range(len(x_axis_values)):
                if int(x_axis_values[j])==i:
                    count += y_axis_fp32_values[j]
        x_fp32.append('['+str(r[0])+","+str(r[-1:][0])+']')
        y_fp32.append(count)
    
    # Get counts for each range in FP64
    x_fp64 = []
    y_fp64 = []
    for r in chunks(list(range(-1023, 1026)), FP64_EXPONENT_SIZE):
        count = 0
        for i in r:
            for j in range(len(x_axis_values)):
                if int(x_axis_values[j])==i:
                    count += y_axis_fp64_values[j]
        x_fp64.append('['+str(r[0])+","+str(r[-1:][0])+']')
        y_fp64.append(count)
    
    # Plot data
    fig, axs = plt.subplots(2, 1)
    axs[0].bar(x_fp64, y_fp64)
    axs[0].xaxis.set_ticks(x_fp64)
    axs[0].tick_params(axis='x', labelrotation=60)
    #axs[0].set_xticklabels(x_fp64, rotation=60)
    axs[0].set_title('FP64')
    
    axs[1].bar(x_fp32, y_fp32, color=(0.2, 0.8, 0.2))
    #axs[1].set_xticklabels(x_fp32, rotation=60)
    axs[1].xaxis.set_ticks(x_fp32)
    axs[1].tick_params(axis='x', labelrotation=60)
    axs[1].set_xlabel('Exponent Range')
    axs[1].set_title('FP32')
    fig.tight_layout()
    #plt.show()
    
    # Save plot
    plt.savefig(destination_directory + '/' + plot_name.replace(" ", "_"))
        
# Generates exponent plots for each source code line recorded in the histogram_data in the
# directory plots_root_path
# Returns: nothing
def histogramPerProgram(plots_root_path, histogram_data):
    createDirectory(plots_root_path)

    accumulated_exponent_dict = {'fp32': {}, 'fp64': {}}
    plot_name = os.path.basename(histogram_data[0]['input'])

    # Looping through each line recorded in histogram json file and accumulating exponent data
    for line_data in histogram_data:
        accumulated_exponent_dict = accumulateOverKey(accumulated_exponent_dict, line_data, 'fp32')
        accumulated_exponent_dict = accumulateOverKey(accumulated_exponent_dict, line_data, 'fp64')

    # Filling missing exponent records with 0s in fp32 and fp64 dictionaries for plotting purposes
    keys_set = mergeKeys(accumulated_exponent_dict, ['fp32', 'fp64'])

    # Saving figure as the input name
    program_plot_path = plots_root_path+'/program'
    createDirectory(program_plot_path)
    plotExponentHistogramRanges(list(keys_set),
                                   list(cleanUpField(accumulated_exponent_dict, 'fp32', keys_set).values()),
                                   list(cleanUpField(accumulated_exponent_dict, 'fp64', keys_set).values()),
                                   program_plot_path,
                                   plot_name + '.png')

    return accumulated_exponent_dict


# Generates exponent plots for each source code line recorded in the histogram_data in the
# directory plots_root_path
# Returns: dict with meta data of plots
def histogramPerFile(plots_root_path, histogram_data):
    createDirectory(plots_root_path)

    accumulated_exponent_dict = {}
    # Looping through each line recorded in histogram json file and accumulating exponent data per file
    for line_data in histogram_data:
        # File name in histogram json is the complete path. Extract the base name.
        file_name = os.path.basename(line_data['file'])

        if file_name not in accumulated_exponent_dict:
            accumulated_exponent_dict[file_name] = {'fp32': {}, 'fp64': {}}

        accumulated_exponent_dict[file_name] = accumulateOverKey(accumulated_exponent_dict[file_name], line_data,
                                                                   'fp32')
        accumulated_exponent_dict[file_name] = accumulateOverKey(accumulated_exponent_dict[file_name], line_data,
                                                                   'fp64')

    plot_meta_data = {} # dictionary with key: plot_name, value: application file
    # Looping through each record which corresponds to a file
    for file_name, file_data in accumulated_exponent_dict.items():
        split_file_name = os.path.splitext(file_name)
        plot_name = split_file_name[0] + split_file_name[1]

        # Filling missing exponent records with 0s in fp32 and fp64 dictionaries for plotting purposes
        keys_set = mergeKeys(file_data, ['fp32', 'fp64'])

        # Saving figure as the source file name
        file_plot_path = plots_root_path+'/files'
        createDirectory(file_plot_path)
        plotExponentHistogramRanges(list(keys_set),
                                       list(cleanUpField(file_data, 'fp32', keys_set).values()),
                                       list(cleanUpField(file_data, 'fp64', keys_set).values()),
                                       file_plot_path,
                                       plot_name + '.png')
        plot_meta_data[plot_name + '.png'] = file_name

    return accumulated_exponent_dict, plot_meta_data


# Generates exponent plots for each source code line recorded in the histogram_data in the
# directory plots_root_path
# Returns: nothing
def histogramPerLine(plots_root_path, histogram_data):
    createDirectory(plots_root_path)

    # Looping through each line recorded in histogram json file
    for line_data in histogram_data:
        # File name in histogram json is the complete path. Extract the base name to create a directory for plots
        # corresponding to that source file
        file_name = os.path.basename(line_data['file'])
        split_file_name = os.path.splitext(file_name)
        plots_for_file_directory = split_file_name[0] + split_file_name[1].split('.')[1].capitalize()

        createDirectory(plots_root_path + '/' + plots_for_file_directory)

        # Filling missing exponent records with 0s in fp32 and fp64 dictionaries for plotting purposes
        keys_set = mergeKeys(line_data, ['fp32', 'fp64'])

        # Plotting histogram as the line number in directory corresponding to source file
        plotExponentHistogram(list(keys_set),
                                list(cleanUpField(line_data, 'fp32', keys_set).values()),
                                list(cleanUpField(line_data, 'fp64', keys_set).values()),
                                plots_root_path + '/' + plots_for_file_directory,
                                str(line_data['line']) + '.png')

    return histogram_data

# Creates the HTML report with plots
# Returns: None
def createReport(report_title, plots_root_path, file_metadata):   
    # Load template
    fd = open(ROOT_REPORT_TEMPLATE, 'r')
    templateLines = fd.readlines()
    fd.close()
    
    # Copy style and other files
    shutil.copy2(ROOT_REPORT_TEMPLATE_DIR+'/sitestyle.css', plots_root_path+'/sitestyle.css')
    if not os.path.exists(plots_root_path+'/icons'):
        shutil.copytree(ROOT_REPORT_TEMPLATE_DIR+'/icons', plots_root_path+'/icons')
        
    # Get program and file plot paths
    files_list = glob.glob(plots_root_path+'/program/*')
    program_plot_path = './program/'+os.path.basename(files_list[0])
    files_list = glob.glob(plots_root_path+'/files/*')
    file_plot_paths = []
    for f in files_list:
        file_plot_paths.append('./files/'+os.path.basename(f))
    
    # Write the report using the template
    report_full_name = plots_root_path+'/'+ROOT_REPORT_NAME 
    fd = open(report_full_name, 'w')
    for i in range(len(templateLines)):
        if '<!-- REPORT_TITLE -->' in templateLines[i]:
            fd.write(report_title+'\n')
        elif '<!-- FPC_PROGRAM_PLOT -->' in templateLines[i]:
            #fd.write('<img src="'+program_plot_path+'" height="400" alt=""/>')
            fd.write('<a href="'+program_plot_path+'"><img src="'+program_plot_path+'" height="400" alt=""/></a>')
        elif '<!-- FPC_FILE_PLOT -->' in templateLines[i]:
            for f in file_plot_paths:
                application_file = file_metadata[os.path.basename(f)]
                fd.write('<tr class="tr_class"> <td class="td_class"> File: '+application_file+' </td></tr>\n')
                #fd.write('<tr class="tr_class"> <td class="td_class"> <img src="'+f+'" height="300" alt=""/></td></tr>\n')
                fd.write('<tr class="tr_class"> <td class="td_class"> <a href="'+f+'"><img src="'+f+'" height="300" alt=""/></a></td></tr>\n')
        else:
          fd.write(templateLines[i])
    fd.close()
    print('Report created: ' + report_full_name)

# Gets the paths for histogram traces
def getHistogramTracePaths(p):
    fileList = []
    for root, dirs, files in os.walk(p):
        for file in files:
            fileName = os.path.split(file)[1]
            if fileName.startswith('histogram_') and fileName.endswith(".json"):
                f = str(os.path.join(root, file))
                fileList.append(f)
    return fileList

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plotting histogram of exponents')
    parser.add_argument('-t', '--title', nargs=1, type=str, help='Title of report.')
    #parser.add_argument('-f', '--json_file',
    #                    help='JSON format histogram file output by FPChecker containing exponent data',
    #                    required=True)
    parser.add_argument('-o', '--output_dir',
                        help='Path to directory to create report in',
                        default='./fpc-report')
    parser.add_argument('dir', nargs='?', default=os.getcwd())
    #parser.add_argument('-l', '--refinement_level',
    #                    help='1: Line level histograms. '
    #                         '2: File level histograms. '
    #                         '3: Full program histogram',
    #                    type=int,
    #                    default=1)
    arguments = parser.parse_args()

    #if arguments.refinement_level == 1:
    #    histogramPerLine(arguments.output_dir, json_data)
    #elif arguments.refinement_level == 2:
    #    histogramPerFile(arguments.output_dir, json_data)
    #elif arguments.refinement_level == 3:
    #    histogramPerProgram(arguments.output_dir, json_data)
    
    # Get paths of histogram traces
    reports_path = arguments.dir
    fileList = getHistogramTracePaths(reports_path)
    print('Trace files found:', len(fileList))
    
    # Create plots
    loadTraces(fileList)
    histogramPerProgram(arguments.output_dir, accumulated_data)
    data, file_metadata = histogramPerFile(arguments.output_dir, accumulated_data)
    
    # Create report
    report_title = ''
    if (arguments.title):
      report_title = arguments.title[0]
    createReport(report_title, arguments.output_dir, file_metadata)
