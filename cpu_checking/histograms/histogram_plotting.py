import argparse
import collections
import glob
import json
import os
import sys
import matplotlib.pyplot as plt


def load_report(file_name):
    f = open(file_name, 'r')
    data = json.load(f)
    f.close()
    return data


# Create a new directory if does not exist
# Returns: nothing
def create_directory(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)


# Accumulates quantity in data dictionary over specified key and adds to init_dict
# init_dict and data need to have same structure
# Returns: dictionary
def accumulate_over_key(init_dict, data, key):
    for exponent in data[key].keys():
        if exponent in init_dict[key]:
            init_dict[key][exponent] += data[key][exponent]
        else:
            init_dict[key][exponent] = data[key][exponent]

    return init_dict


# Merging keys from selected fields of a dictionary.
# Returns: set
def merge_keys(data_dictionary, fields_to_merge_from):
    keys_set = set()
    for field in fields_to_merge_from:
        keys_set.update(data_dictionary[field].keys())

    return keys_set


# Initializing missing values from keys_set in field to 0
# Returns: dictionary
def clean_up_field(data_dictionary, field, keys_set):
    cleaned_up_field = dict.fromkeys(keys_set, 0)
    cleaned_up_field.update(data_dictionary[field])
    return cleaned_up_field


# Multibar plotting of exponent histogram
# Returns: nothing
def plot_exponent_histogram(x_axis_values, y_axis_fp32_values, y_axis_fp64_values, destination_directory, plot_name):
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


# Generates exponent plots for each source code line recorded in the histogram_data in the
# directory plots_root_path
# Returns: nothing
def histogram_per_program(plots_root_path, histogram_data):
    create_directory(plots_root_path)

    accumulated_exponent_dict = {'fp32': {}, 'fp64': {}}
    plot_name = os.path.basename(histogram_data[0]['input'])

    # Looping through each line recorded in histogram json file and accumulating exponent data
    for line_data in histogram_data:
        accumulated_exponent_dict = accumulate_over_key(accumulated_exponent_dict, line_data, 'fp32')
        accumulated_exponent_dict = accumulate_over_key(accumulated_exponent_dict, line_data, 'fp64')

    # Filling missing exponent records with 0s in fp32 and fp64 dictionaries for plotting purposes
    keys_set = merge_keys(accumulated_exponent_dict, ['fp32', 'fp64'])

    return [keys_set,
            clean_up_field(accumulated_exponent_dict, 'fp32', keys_set),
            clean_up_field(accumulated_exponent_dict, 'fp64', keys_set),
            plots_root_path,
            plot_name + '.png']

    # Saving figure as the input name
    plot_exponent_histogram(list(keys_set),
                            list(clean_up_field(accumulated_exponent_dict, 'fp32', keys_set).values()),
                            list(clean_up_field(accumulated_exponent_dict, 'fp64', keys_set).values()),
                            plots_root_path,
                            plot_name + '.png')

    return accumulated_exponent_dict


# Generates exponent plots for each source code line recorded in the histogram_data in the
# directory plots_root_path
# Returns: nothing
def histogram_per_file(plots_root_path, histogram_data):
    create_directory(plots_root_path)

    accumulated_exponent_dict = {}
    # Looping through each line recorded in histogram json file and accumulating exponent data per file
    for line_data in histogram_data:
        # File name in histogram json is the complete path. Extract the base name.
        file_name = os.path.basename(line_data['file'])

        if file_name not in accumulated_exponent_dict:
            accumulated_exponent_dict[file_name] = {'fp32': {}, 'fp64': {}}

        accumulated_exponent_dict[file_name] = accumulate_over_key(accumulated_exponent_dict[file_name], line_data,
                                                                   'fp32')
        accumulated_exponent_dict[file_name] = accumulate_over_key(accumulated_exponent_dict[file_name], line_data,
                                                                   'fp64')


    # Looping through each record which corresponds to a file
    for file_name, file_data in accumulated_exponent_dict.items():
        split_file_name = os.path.splitext(file_name)
        plot_name = split_file_name[0] + split_file_name[1].split('.')[1].capitalize()

        # Filling missing exponent records with 0s in fp32 and fp64 dictionaries for plotting purposes
        keys_set = merge_keys(file_data, ['fp32', 'fp64'])

        return [keys_set,
                clean_up_field(file_data, 'fp32', keys_set),
                clean_up_field(file_data, 'fp64', keys_set),
                plots_root_path,
                plot_name + '.png']

        # Saving figure as the source file name
        plot_exponent_histogram(list(keys_set),
                                list(clean_up_field(file_data, 'fp32', keys_set).values()),
                                list(clean_up_field(file_data, 'fp64', keys_set).values()),
                                plots_root_path,
                                plot_name + '.png')

    return accumulated_exponent_dict


# Generates exponent plots for each source code line recorded in the histogram_data in the
# directory plots_root_path
# Returns: nothing
def histogram_per_line(plots_root_path, histogram_data):
    create_directory(plots_root_path)

    # Looping through each line recorded in histogram json file
    for line_data in histogram_data:
        # File name in histogram json is the complete path. Extract the base name to create a directory for plots
        # corresponding to that source file
        file_name = os.path.basename(line_data['file'])
        split_file_name = os.path.splitext(file_name)
        plots_for_file_directory = split_file_name[0] + split_file_name[1].split('.')[1].capitalize()

        create_directory(plots_root_path + '/' + plots_for_file_directory)

        # Filling missing exponent records with 0s in fp32 and fp64 dictionaries for plotting purposes
        keys_set = merge_keys(line_data, ['fp32', 'fp64'])

        return [keys_set,
                clean_up_field(line_data, 'fp32', keys_set),
                clean_up_field(line_data, 'fp64', keys_set),
                plots_root_path,
                plot_name + '.png']

        # Plotting histogram as the line number in directory corresponding to source file
        plot_exponent_histogram(list(keys_set),
                                list(clean_up_field(line_data, 'fp32', keys_set).values()),
                                list(clean_up_field(line_data, 'fp64', keys_set).values()),
                                plots_root_path + '/' + plots_for_file_directory,
                                str(line_data['line']) + '.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plotting histogram of exponents')
    parser.add_argument('-f', '--json_file',
                        help='JSON format histogram file output by FPChecker containing exponent data',
                        required=True)
    parser.add_argument('-o', '--output_dir',
                        help='Path to directory to create plots in',
                        default='plots')
    parser.add_argument('-l', '--refinement_level',
                        help='1: Line level histograms. '
                             '2: File level histograms. '
                             '3: Full program histogram',
                        type=int,
                        default=1)
    arguments = parser.parse_args()

    json_data = load_report(arguments.json_file)

    # json_formatted_obj = json.dumps(histogram_data, indent=2)
    # print(json_formatted_obj)

    if arguments.refinement_level == 1:
        histogram_per_line(arguments.output_dir, json_data)
    elif arguments.refinement_level == 2:
        histogram_per_file(arguments.output_dir, json_data)
    elif arguments.refinement_level == 3:
        histogram_per_program(arguments.output_dir, json_data)
