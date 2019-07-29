
# import python wrapper for or-tools CP-SAT sovler.
from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import getopt
import sys
from random import *


def Vis_Gantt_Chart(input_df, File_Path, file_name):
    """Draw the gantt chart based on the results"""
    #fix the random seed
    seed(1)
    ##get the random colours for each crew
    get_colors = lambda n: list(
    map(lambda i: "#" + "%06x" % randint(0, 0xFFFFFF),range(n))
    )
    ## %06x means 6 digits and x for hexidcimal
    line_list = set(input_df['line'])
    color_list = get_colors(len(line_list))

    color = dict(zip(line_list, color_list)) ##zip the color and crew into a map

    #color = {'c1' : 'turquoise', 'c2' : 'crimson'}
    hatch_dic = {'produce' : '//'}
    fig,ax=plt.subplots(figsize=(10,9))

    ylabels=[]

    for i, task in enumerate(input_df.groupby('specification')):
        ylabels.append(task[0]) #get the labels for the y axis

        for crew_group in task[1].groupby('line'):

            data = crew_group[1][['start_time', 'task_duration']]


            for type_group in crew_group[1].groupby('task'):
                ax.broken_barh(
                data.values, (i*10, 9), facecolor = color[crew_group[0]],
                hatch = hatch_dic[type_group[0]]
                ) #draw the bar chart


    ax.set_yticks(range(5, len(ylabels)*10+5, 10))
    ax.set_yticklabels(ylabels)
    ax.set_xlabel('time')
    plt.tight_layout()

    ## add the legend ##
    patches1 = [
    mpatches.Patch(color = color, label = key) for (key, color) in color.items()
    ]
    patches2 = [
    mpatches.Patch(facecolor = 'white', hatch = hatch, label = key)
    for (key, hatch) in hatch_dic.items()
    ]

    legend1 = plt.legend(handles = patches1, bbox_to_anchor = (0, 1))
    plt.legend(handles = patches2, loc = 'best', fontsize = 12)

    plt.gca().add_artist(legend1)

    plt.title('Harvesting Schedule', loc = 'center')

    figure_saving_path = File_Path + '/' + 'Figure_No_' + file_name + '.png'
    plt.savefig(figure_saving_path)


    plt.show()


def create_data_model(file_path):
    """Read data from excels"""
    # changeover dataframe

    change_over_df = pd.read_excel(file_path + '/Schedule_CO_time.xlsx',
    index_col = 0)
    change_over_matrix = change_over_df.to_numpy().tolist()

    # time window dataframe
    time_window_df = pd.read_excel(file_path + '/Schedule_Due_time.xlsx',
    index_col = 0)
    # convert hrs to minutes
    time_window_df.loc[:,:] *= 60

    due_time_list = time_window_df['Due_time (hours)'].tolist()
    time_windows_list = [(0, a) for a in due_time_list]

    # num of lines
    num_line = 3

    # starting product for each line
    start_product = [0, 1, 2]

    # finishing product for each line
    # here a dummy node is introduced
    finish_product = [change_over_df.shape[0] - 1 for i in range(len(start_product))] #[70, 70, 70]

    # run time of each product
    run_time_df = pd.read_excel(file_path + '/Schedule_Run_length.xlsx',
    index_col = 0)
    run_time_list = run_time_df ['Run_length (minutes)'].tolist()

    # line constraints for products
    line_constraint_df = pd.read_excel(
    file_path + '/Schedule_Line_constraint.xlsx',
    index_col = 0)
    # convert to preliminary dict
    preliminary_dict = line_constraint_df.to_dict()
    # convert preliminary dict to the format needed
    # {product : [line1, Line2....]}

    line_constraint_dict = {i :[] for i in range(line_constraint_df.shape[0])}
    for key, key_values in preliminary_dict.items():
        for secondary_key, values in key_values.items():
            if values == 1:
                line_value = int(key.split('L')[1]) - 1
                line_constraint_dict[secondary_key - 1].append(line_value)

    # construction of the data
    data = {}

    data['change_over_matrix'] = change_over_matrix
    data['time_windows'] = time_windows_list
    data['num_lines'] = num_line
    data['starts'] = start_product
    data['ends'] = finish_product
    data['product_production_time'] = run_time_list
    data['line_constraint'] = line_constraint_dict
    #print(data['change_over_matrix'])
    return data

def print_solution(data, manager, scheduling, solution):
    """Prints solution on console."""
    max_production_time = 0
    for line_id in range(data['num_lines']):
        index = scheduling.Start(line_id)
        plan_output = 'Production schedule for line {}:\n'.format(line_id)
        production_time = 0
        while not scheduling.IsEnd(index):
            plan_output += ' {} -> '.format(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(scheduling.NextVar(index))
            production_time += scheduling.GetArcCostForVehicle(
                previous_index, index, line_id)
        plan_output += '{}\n'.format(manager.IndexToNode(index))
        plan_output += \
        'Total operation time of line: {}m\n'.format(production_time)
        print(plan_output)
        max_production_time = max(production_time, max_production_time)
    print('Maximum of the route distances: {}m'.format(max_production_time))

def get_schedule(data, manager, scheduling, solution):
    """Get the schedule to a dataframe"""
    # define the list that is to be converted to df
    schedule_list_sum = []

    for line_id in range(data['num_lines']):
        index = scheduling.Start(line_id)
        # initialise the start and finish time
        # of each product at each line
        start_time = 0
        finish_time = 0
        while not scheduling.IsEnd(index):
            previous_index = index
            index = solution.Value(scheduling.NextVar(index))
            production_time = scheduling.GetArcCostForVehicle(
                previous_index, index, line_id)
            if index >= len(data['change_over_matrix']):
                index = len(data['change_over_matrix']) - 1


            o = 'produce'
            l = line_id
            p = previous_index + 1
            d = data['product_production_time'][previous_index]

            schedule_list_sum.append([o, l, start_time, finish_time, p, d])

            start_time += production_time
            finish_time = start_time + \
                data['product_production_time'][index]

    # sort the list based on start time
    schedule_list_sum.sort(key = lambda x : x[2])

    schedule_df = pd.DataFrame(schedule_list_sum)
    schedule_df.columns = ['task', 'line', 'start_time', 'finish_time',
                        'product', 'task_duration']
    schedule_df.loc[:, 'specification'] = 'Line ' + \
    schedule_df['line'].astype(str) + ' ' + 'produces ' + \
    schedule_df['product'].astype(str)


    return schedule_df

def main():
    """Entry point of the program."""

    opts, args = getopt.gnu_getopt(sys.argv[1:], '',
    ['file_path='])
    # process all the options
    for opt, arg in opts :
        if opt == '--file_path':
            file_path = arg
    # Input the data.
    data = create_data_model(file_path)

    # Create the routing/scheduling index manager.

    manager = pywrapcp.RoutingIndexManager(
        len(data['change_over_matrix']), data['num_lines'], data['starts'],
        data['ends'])

    # Create Routing Model.
    scheduling = pywrapcp.RoutingModel(manager)

    # Create and register a transit/schedule callback
    def time_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from scheduling variable Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['change_over_matrix'][from_node][to_node] + \
                data['product_production_time'][from_node]

    schedule_callback_index = scheduling.RegisterTransitCallback(time_callback)

    # Define cost of each arc.
    scheduling.SetArcCostEvaluatorOfAllVehicles(schedule_callback_index)

    # Add Time Windows constraint.
    time = 'Time'
    scheduling.AddDimension(
        schedule_callback_index,
        0,  # allow waiting time
        12000,  # maximum time per vehicle
        True,  # Don't force start cumul to zero.
        time)
    time_dimension = scheduling.GetDimensionOrDie(time)
    # The method SetGlobalSpanCostCoefficient sets a large coefficient (100)
    # for the global span of the routes, which in this example is the
    # maximum of the distances of the routes. This makes the global span
    # the predominant factor in the objective function, so the program
    # minimizes the length of the longest route.
    time_dimension.SetGlobalSpanCostCoefficient(100)

    # Add time window constraints for each location except depot.
    for product, time_window in enumerate(data['time_windows']):
        if product == 0 or 1 or 2 or len(data['time_windows']) - 1:
            continue
        print(time_dimension.CumulVar(product))
        time_dimension.CumulVar(product).SetRange(time_window[0],
                                                    time_window[1])


    # define the lines that can produce certain product

    for product, line_list in data['line_constraint'].items():
        allowed_line_list = [-1]
        allowed_index = manager.NodeToIndex(product)
        allowed_line_list.extend(line_list)

        scheduling.VehicleVar(product).SetValues(allowed_line_list)


    # Instantiate route start and end times to produce feasible times.
    for i in range(data['num_lines']):
        scheduling.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(scheduling.Start(i)))
        scheduling.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(scheduling.End(i)))

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH)

    # search_parameters.first_solution_strategy = (
    #     routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    search_parameters.time_limit.seconds = 30
    search_parameters.log_search = True
    # get the attribute of the search_parameters
    #print(dir(search_parameters))

    # Solve the problem.
    assignment = scheduling.SolveWithParameters(search_parameters)

    # Print solution on console.
    if assignment:
        #file_path = 'Y:\\Pyomo_projects\\PnG_CP'
        figure_name = 'Demo_figure_heuristic'
        excel_name = '/' + 'Detailed_schedule_heuristic.xlsx'
        excel_path = file_path + excel_name
        input_df = get_schedule(data, manager, scheduling, assignment)
        writer_reference = pd.ExcelWriter(excel_path)
        input_df.to_excel(writer_reference, startcol = 4, startrow = 1)
        worksheet = writer_reference.sheets['Sheet1']
        worksheet.write_string(10, 0, 'Detailed Schedule')
        writer_reference.save()

        Vis_Gantt_Chart(input_df, file_path, figure_name)
        # print_solution(data, manager, scheduling, assignment)
    else:
        print('No solution found !')


if __name__ == '__main__':
    main()
