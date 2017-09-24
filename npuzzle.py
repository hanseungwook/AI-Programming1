"""
COMS W4701 Artificial Intelligence - Programming Homework 1

In this assignment you will implement and compare different search strategies
for solving the n-Puzzle, which is a generalization of the 8 and 15 puzzle to
squares of arbitrary size (we will only test it with 8-puzzles for now). 
See Courseworks for detailed instructions.

@author: Seungwook Han (sh3264)
"""

import time

def state_to_string(state):
    row_strings = [" ".join([str(cell) for cell in row]) for row in state]
    return "\n".join(row_strings)


# Returns a new state with the cells (i1,j1) and (i2,j2) swapped. 

def swap_cells(state, i1, j1, i2, j2):

    value1 = state[i1][j1]
    value2 = state[i2][j2]
    
    new_state = []
    for row in range(len(state)): 
        new_row = []
        for column in range(len(state[row])): 
            if row == i1 and column == j1: 
                new_row.append(value2)
            elif row == i2 and column == j2:
                new_row.append(value1)
            else: 
                new_row.append(state[row][column])
        new_state.append(tuple(new_row))
    return tuple(new_state)
    

# This function returns a list of possible successor states resulting
# from applicable actions. 
# The result should be a list containing (Action, state) tuples. 
# For example [("Up", ((1, 4, 2),(0, 5, 8),(3, 6, 7))), 
#              ("Left",((4, 0, 2),(1, 5, 8),(3, 6, 7)))] 

def get_successors(state):

    hole = ()
    holeRow = 0
    holeElement = 0

    # Saves the row and column number of the hole '0' into holeRow and hole Element respectively
    for row in range(len(state)):
        for element in range(len(state[row])):
            if (state[row][element] == 0):
                holeRow = row
                holeElement = element
  
    # Initializing the child_states list and tuples for each of the actions
    child_states = []
    left = ()
    right = ()
    up = ()
    down = ()

    # Swapping accordingly to the action if it is possible
    if ((holeElement % 3) != 2):
        left = ('Left', swap_cells(state, holeRow, holeElement, holeRow, (holeElement + 1)))
    if ((holeElement % 3) != 0):
        right = ('Right', swap_cells(state, holeRow, holeElement, holeRow, (holeElement - 1)))
    if (holeRow != 2):
        up = ('Up', swap_cells(state, holeRow, holeElement, (holeRow + 1), holeElement))
    if (holeRow != 0):
        down = ('Down', swap_cells(state, holeRow, holeElement, (holeRow - 1), holeElement))

    # If the respective tuples are not empty, append them to the child_states list in order
    # of Left -> Right -> Up -> Down
    if (left):
        child_states.append(left)
    if (right):
        child_states.append(right)
    if (up):
        child_states.append(up)
    if (down):
        child_states.append(down)
    
    return child_states


# Returns True if the state is a goal state, False otherwise.         

def goal_test(state):

    goalState = ((0,1,2), (3,4,5), (6,7,8))
    if (state == goalState):
        return True
    else:
        return False


# Function that uses the parents and actions dictionaries
# to put the solution of actions in order as a list

def get_solution(initialState, parents, actions):
    solution = []
    state = ((0, 1, 2),
             (3, 4, 5),
             (6, 7, 8))

    # Until we back track from the goal to the initial state
    # use the actions dictionary to get the actions and add it
    # to the front of the solution list and set the parent state
    # as the state to continue the while loop 
    while (parents.get(state) != initialState):
        solution.insert(0, actions.get(state))
        state = parents.get(state)

    solution.insert(0, actions.get(state))

    return solution


# Breadth first search.
# Returns three values: A list of actions, the number of states expanded, and
# the maximum size of the frontier.  

def bfs(state):

    # Initializing the sets, dictionaries, and variables
    parents = {}
    actions = {}

    states_expanded = 0
    max_frontier = 0
    initialState = state

    frontier = [state]
    explored = set()
    seen = set()    
    seen.add(state)

    while (frontier):
        # Pop the the first element in the queue
        currentState = frontier.pop(0)
        explored.add(currentState)
        states_expanded += 1

        # Test if the current state is the goal
        if (goal_test(currentState)):
            solution = get_solution(initialState, parents, actions)
            return solution, states_expanded, max_frontier

        # Get list of possible successors from the current state
        successors = get_successors(currentState)
        for successor in successors:
            # If the successor was not saved in the parents and actions dictionaries
            # save the respective key and values
            childState = successor[1]
            childAction = successor[0]
            if (parents.get(childState) == None):
                parents[childState] = currentState
            if (actions.get(childState) == None):
                actions[childState] = childAction

            # If child state not in seen and explored, add to the back of the frontier
            # queue and add to seen set
            if ((childState not in seen) and (childState not in explored)):
                frontier.append(childState)
                seen.add(childState)
                if (len(frontier) > max_frontier):
                    max_frontier = len(frontier)

    return None, states_expanded, max_frontier


# Depth first search.
# Returns three values: A list of actions, the number of states expanded, and
# the maximum size of the frontier.  

def dfs(state):

    # Initializing the sets, dictionaries, and variables
    parents = {}
    actions = {}

    states_expanded = 0
    max_frontier = 0
    initialState = state            

    frontier = [state]
    explored = set()
    seen = set()    
    seen.add(state)

    # The only difference from the bfs algorithm is that the state to explore
    # is popped from the back of the frontier list - implementing a stack rather
    # than a queue
    while (frontier):
        # Pop from the end of the list (as a stack)
        currentState = frontier.pop()
        explored.add(currentState)
        states_expanded += 1

        if (goal_test(currentState)):
            solution = get_solution(initialState, parents, actions)
            return solution, states_expanded, max_frontier

        # Get list of possible successors from the current state
        successors = get_successors(currentState)
        for successor in successors:
            # If the successor was not saved in the parents and actions dictionaries
            # save the respective key and values
            childState = successor[1]
            childAction = successor[0]
            if (parents.get(childState) == None):
                parents[childState] = currentState
            if (actions.get(childState) == None):
                actions[childState] = childAction

            # If child state not in seen and explored, add to the back of the frontier
            # stack and add to seen set
            if ((childState not in seen) and (childState not in explored)):
                frontier.append(childState)
                seen.add(childState)
                if (len(frontier) > max_frontier):
                    max_frontier = len(frontier)

    return None, states_expanded, max_frontier

    
# Returns the number of misplaced tiles.
    
def misplaced_heuristic(state):

    misplaced = 0

    # Excludes 0 from the count for misplaced tiles and counts the others
    for row in range(len(state)):
        for element in range(len(state[row])):
            if (state[row][element] == 0):
                continue
            if (state[row][element] != row * 3 + element):
                misplaced += 1
                
    return misplaced


# For each misplaced tile, compute the manhattan distance between the current
# position and the goal position. THen sum all distances. 

def manhattan_heuristic(state):

    manhattanDist = 0

    # Iterate through the whole state one by one (excluding 0)
    # and calculate the row/column (x/y) distances by using
    # the remainder the division values. 
    for row in range(len(state)):
        for column in range(len(state[row])):
            if (state[row][column] == 0):
                continue
            else:
                tile = int(state[row][column])
                xDist = abs(column - (tile % 3))
                yDist = abs(row - int(tile / 3))
                totalDist = xDist + yDist
                manhattanDist += totalDist

    return manhattanDist


# Breadth first search using the heuristic function passed as a parameter.
# Returns three values: A list of actions, the number of states expanded, and
# the maximum size of the frontier.  

def best_first(state, heuristic = misplaced_heuristic):

    from heapq import heappush
    from heapq import heappop

    # Initializing the sets, dictionaries, priority queue and variables
    parents = {}
    actions = {}
    frontier = []
    
    states_expanded = 0
    max_frontier = 0
    initialState = state
        
    heappush(frontier, (0, state))
    explored = set()
    seen = set()    
    seen.add(state)

    while (frontier):
        # Since the frontier list is organized as [(cost, state), ...], the second
        # element in the list tuple is the current state
        currentState = heappop(frontier)[1]
        explored.add(currentState)
        states_expanded += 1

        # Test if the current state is at goal and, if so, get solution
        if (goal_test(currentState)):
            solution = get_solution(initialState, parents, actions)
            return solution, states_expanded, max_frontier

        # Get a list of successors of the current state
        successors = get_successors(currentState)
        for successor in successors:
            # If the successor was not saved in the parents and actions dictionaries
            # save the respective key and values
            childState = successor[1]
            childAction = successor[0]
            if (parents.get(childState) == None):
                parents[childState] = currentState
            if (actions.get(childState) == None):
                actions[childState] = childAction

            # If child state not in seen and explored, push to the frontier
            # priority queue and add to seen set
            if ((childState not in seen) and (childState not in explored)):
                childCost = heuristic(childState)
                heappush(frontier, (childCost, childState))
                seen.add(childState)
                if (len(frontier) > max_frontier):
                    max_frontier = len(frontier)

    return None, 0, 0 


# A-star search using the heuristic function passed as a parameter. 
# Returns three values: A list of actions, the number of states expanded, and
# the maximum size of the frontier.  
def astar(state, heuristic = misplaced_heuristic):

    from heapq import heappush
    from heapq import heappop

    # Initializing the dictionaries, sets, priority queue and variables
    parents = {}
    actions = {}
    costs = {}
    costs[state] = 0
    frontier = []
   
    states_expanded = 0
    max_frontier = 0
    initialState = state
        
    heappush(frontier, (0, state))
    explored = set()
    seen = set()    
    seen.add(state)

    while (frontier):
        # Since the frontier list is organized as [(cost, state), ...], the second
        # element in the list tuple is the current state
        currentState = heappop(frontier)[1]
        explored.add(currentState)
        states_expanded += 1

        # Test if the current state is at goal and, if so, get solution
        if (goal_test(currentState)):
            solution = get_solution(initialState, parents, actions)
            return solution, states_expanded, max_frontier

        # Get a list of successors of the current state
        successors = get_successors(currentState)
        for successor in successors:
            # If the successor was not saved in the parents and actions dictionaries
            # save the respective key and values
            childState = successor[1]
            childAction = successor[0]
            if (parents.get(childState) == None):
                parents[childState] = currentState
            if (actions.get(childState) == None):
                actions[childState] = childAction

            # If child state not in seen and explored, push to the frontier
            # priority queue and add to seen set
            if ((childState not in seen) and (childState not in explored)):
                # Calculating the heuristic cost of the child + the cost of the path until the state
                costs[childState] = costs.get(currentState) + 1
                childCost = heuristic(childState) + costs.get(childState)
                heappush(frontier, (childCost, childState))
                seen.add(childState)
                if (len(frontier) > max_frontier):
                    max_frontier = len(frontier)

    return None, states_expanded, max_frontier # No solution found

    
# Helper function to format test output. 
def print_result(solution, states_expanded, max_frontier):
    if solution is None: 
        print("No solution found.")
    else: 
        print("Solution has {} actions.".format(len(solution)))
    print("Total states expanded: {}.".format(states_expanded))
    print("Max frontier size: {}.".format(max_frontier))

if __name__ == "__main__":

    #Easy test case
    test_state = ((1, 4, 2),
                  (0, 5, 8), 
                  (3, 6, 7))  

    #More difficult test case
    test_state = ((7, 2, 4),
                 (5, 0, 6), 
                 (8, 3, 1))  

    print(state_to_string(test_state))
    print()

    print("====BFS====")
    start = time.time()
    solution, states_expanded, max_frontier = bfs(test_state) #
    end = time.time() 
    print_result(solution, states_expanded, max_frontier)
    if solution is not None:
        print(solution)
    print("Total time: {0:.3f}s".format(end-start))

    print() 
    print("====DFS====") 
    start = time.time()
    solution, states_expanded, max_frontier = dfs(test_state)
    end = time.time()
    print_result(solution, states_expanded, max_frontier)
    print("Total time: {0:.3f}s".format(end-start))

    print() 
    print("====Greedy Best-First (Misplaced Tiles Heuristic)====") 
    start = time.time()
    solution, states_expanded, max_frontier = best_first(test_state, misplaced_heuristic)
    end = time.time()
    print_result(solution, states_expanded, max_frontier)
    print("Total time: {0:.3f}s".format(end-start))
    
    print() 
    print("====A* (Misplaced Tiles Heuristic)====") 
    start = time.time()
    solution, states_expanded, max_frontier = astar(test_state, misplaced_heuristic)
    end = time.time()
    print_result(solution, states_expanded, max_frontier)
    print("Total time: {0:.3f}s".format(end-start))

    print() 
    print("====A* (Total Manhattan Distance Heuristic)====") 
    start = time.time()
    solution, states_expanded, max_frontier = astar(test_state, manhattan_heuristic)
    end = time.time()
    print_result(solution, states_expanded, max_frontier)
    print("Total time: {0:.3f}s".format(end-start))

