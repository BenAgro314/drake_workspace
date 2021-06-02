from ompl import base as ob
from ompl import geometric as og

def isStateValid(state):
    print(state[0][0])
    print(state[1][0])
    return True

def plan():

    dim = 2
    space = ob.CompoundStateSpace()
    space.addSubspace(ob.RealVectorStateSpace(3), 1)
    space.addSubspace(ob.RealVectorStateSpace(3), 0.1)
    bounds = ob.RealVectorBounds(3)
    bounds.setLow(-10)
    bounds.setHigh(10)
    space.getSubspace(0).setBounds(bounds)
    bounds = ob.RealVectorBounds(3)
    bounds.setLow(-3.1415)
    bounds.setHigh(3.1415)
    space.getSubspace(1).setBounds(bounds)

    ss = og.SimpleSetup(space)
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))
    state = ob.State(space)

    start = ob.State(space)
    print(start)
    start[0] = 0.5
    start[1] = 4.56e-18
    start[2] = 0.5
    start[3] = 8.7e-18
    start[4] = 1 
    start[5] = 6.16e-17
    print(start)

    goal = ob.State(space)
    goal[0] = 0.6
    goal[1] = 3.18e-18
    goal[2] = 0.2694
    goal[3] = -6.12e-17
    goal[4] = 1
    goal[5] = -6.12e-17
    print(goal)

    ss.setStartAndGoalStates(start, goal)
    print(ss)

    solved = ss.solve()

    if (solved):
        ss.simplifySolution()
        path = ss.getSolutionPath() 
        print(len(path.getStates()))
        for state in path.getStates():
            print(state)

if __name__ == "__main__":
    plan()

