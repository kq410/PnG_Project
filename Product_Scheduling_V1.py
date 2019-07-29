
# import python wrapper for or-tools CP-SAT sovler.
from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def create_data_model():
    """Stores the data for the problem."""

    data = {}

    data['change_over_matrix'] = [
    [
        0, 10, 20, 30, 40, 50, 0
    ],
    [
        10, 0, 10, 20, 30, 40, 0
    ],
    [
        20, 10, 0, 10, 20, 30, 0
    ],
    [
        30, 20, 10, 0, 10, 20, 0
    ],
    [
        40, 30, 20, 10, 0, 10, 0
    ],
    [
        50, 40, 30, 20, 10, 0, 0
    ],
    [
        0, 0, 0, 0, 0, 0, 0
    ]
    ]


    data['time_windows'] = [
        (0, 300),  # depot product
        (0, 300),  # 1
        (0, 300),  # 2
        (0, 300),  # 3
        (0, 300),  # 4
        (0, 300),  # 5
        (0, 300),  # 6
    ]

    data['num_lines'] = 2
    #data['starts'] = 0
    data['starts'] = [0, 1]
    data['ends'] = [6, 6]
    data['product_production_time'] = [30, 20, 50, 60, 30, 25, 0]

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
        1000,  # maximum time per vehicle
        True,  # Don't force start cumul to zero.
        time)
    time_dimension = scheduling.GetDimensionOrDie(time)


    # Add time window constraints for each location except depot.
    for location_idx, time_window in enumerate(data['time_windows']):
        '''if location_idx == 0:
            continue'''
        time_dimension.CumulVar(location_idx).SetRange(time_window[0],
                                                        time_window[1])


    # define the lines that can produce certain product
    for i in [4, 5]:

        allowed_index = manager.NodeToIndex(i)
        print('=========================check out ===================')
        print(allowed_index, i)
        scheduling.VehicleVar(allowed_index).SetValues([-1, 0])

    for i in [2, 3]:
        allowed_index2 = manager.NodeToIndex(i)
        scheduling.VehicleVar(allowed_index2).SetValues([-1, 1])


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
