import glob
import json
import os
import sys
import matplotlib.pyplot as plt


def load_report(file_name):
    f = open(fileName, 'r')
    data = json.load(f)
    f.close()
    return data


# Generates exponent plots for each source code line recorded in the histogram_data in the
# directory plots_root_path
def histogram_per_line(plots_root_path, histogram_data):
    # Create a new directory if does not exist
    if not os.path.isdir(plots_root_path):
        os.mkdir(plots_root_path)

    # Looping through each line recorded in histogram json file
    for line_data in histogram_data:
        # File name in histogram json is the complete path. Extract the base name to create a directory for plots
        # corresponding to that source file
        file_name = os.path.basename(line_data['file'])
        split_file_name = os.path.splitext(file_name)
        plots_for_file_directory = split_file_name[0] + split_file_name[1].split('.')[1].capitalize()

        if not os.path.isdir(plots_root_path + plots_for_file_directory):
            os.mkdir(plots_root_path + plots_for_file_directory)
        plots_for_file_directory += '/'

        # Filling missing exponent records with 0s in fp32 and fp64 dictionaries for plotting purposes.
        new_keys_set = set(line_data['fp32'].keys())
        new_keys_set.update(line_data['fp64'].keys())
        fp32_dictionary = dict.fromkeys(new_keys_set, 0)
        fp64_dictionary = dict.fromkeys(new_keys_set, 0)
        fp32_dictionary.update(line_data['fp32'])
        fp64_dictionary.update(line_data['fp64'])

        # Data gathered for plotting
        x_axis_values = list(fp32_dictionary.keys())
        x_axis_label_pos = list(range(len(x_axis_values)))
        y_axis_fp32_values = list(fp32_dictionary.values())
        y_axis_fp64_values = list(fp64_dictionary.values())

        # print(x_axis_values)
        # print(y_axis_values)

        # Multibar plotting begins
        plt.clf()
        plt.xticks(x_axis_label_pos, x_axis_values)

        x_axis_label_pos[:] = [number - 0.2 for number in x_axis_label_pos]
        plt.bar(x_axis_label_pos, y_axis_fp32_values, 0.4, label="fp32")

        x_axis_label_pos[:] = [number + 0.4 for number in x_axis_label_pos]
        plt.bar(x_axis_label_pos, y_axis_fp64_values, 0.4, label="fp64")

        plt.legend()
        plt.xlabel('Exponent')
        plt.ylabel('Counts')

        # Saving figure as the line number in directory corresponding to source file
        plt.savefig(plots_root_path + plots_for_file_directory + str(line_data['line']) + '.png')


if __name__ == '__main__':
    fileName = sys.argv[1]
    plotsRootPath = sys.argv[2] + '/'
    histogramData = load_report(fileName)

    # json_formatted_obj = json.dumps(histogramData, indent=2)
    # print(histogramData)

    histogram_per_line(plotsRootPath, histogramData)
