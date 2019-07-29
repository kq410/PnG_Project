
# import python wrapper for or-tools CP-SAT sovler.
from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import pandas as pd
import numpy as np

def create_data_model():
    """Read data from excels"""
    # changeover dataframe
    change_over_df = pd.read_excel('Schedule1_CO_time.xlsx', index_col = 0)
    change_over_matrix = change_over_df.to_numpy().tolist()

    # time window dataframe
    time_window_df = pd.read_excel('Schedule1_Due_time.xlsx', index_col = 0)
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
    finish_product = [70, 70, 70]

    # run time of each product
    run_time_df = pd.read_excel('Schedule1_Run_length.xlsx', index_col = 0)
    run_time_list = run_time_df ['Run_length (minutes)'].tolist()

    # line constraints for products
    line_constraint_df = pd.read_excel('Schedule1_Line_constraint.xlsx',
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

    return data

def print_solution(data, manager, scheduling, solution):
    """Prints solution on console."""
    max_production_time = 0
    for line_id in range(data['num_lines']):
        index = scheduling.Start(line_id)
        plan_output = 'Route for vehicle {}:\n'.format(line_id)
        production_time = 0
        while not scheduling.IsEnd(index):
            plan_output += ' {} -> '.format(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(scheduling.NextVar(index))
            production_time += scheduling.GetArcCostForVehicle(
                previous_index, index, line_id)
        plan_output += '{}\n'.format(manager.IndexToNode(index))
        plan_output += 'Distance of the route: {}m\n'.format(production_time)
        print(plan_output)
        max_production_time = max(production_time, max_production_time)
    print('Maximum of the route distances: {}m'.format(max_production_time))


def main():
    """Entry point of the program."""
    # Input the data.
    data = create_data_model()

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
        6000,  # maximum time per vehicle
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
        if product == 0 or 1 or 2 or 70:
            continue
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
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    assignment = scheduling.SolveWithParameters(search_parameters)

    # Print solution on console.
    if assignment:
        print_solution(data, manager, scheduling, assignment)
    else:
        print('No solution found !')


if __name__ == '__main__':
    main()
