import ompl
from ompl import base as ob
from ompl import geometric as og


param = 0.6

class CustomStateSpace(ob.RealVectorStateSpace):

    def __init__(self, nq):
        super().__init__(nq)

    def distance(self, state1, state2):
        return 1


def isStateValid(state):
    return True

def plan():

    dim = 2
    space = ob.RealVectorStateSpace(dim)
    bounds = ob.RealVectorBounds(dim)
    bounds.setLow(-10)
    bounds.setHigh(10)
    space.setBounds(bounds)

    ss = og.SimpleSetup(space)
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))
    state = ob.State(space)
    n = ompl.datastructures.NearestNeighbours()

    start = ob.State(space)
    start[0] = 0
    start[1] = 0
    print(start)

    goal = ob.State(space)
    goal[0] = 1
    goal[1] = 1
    print(goal)
    print(f"distance: {space.distance(start.get(), goal.get())}")

    ss.setStartAndGoalStates(start, goal)

    solved = ss.solve()
    print(ss.getPlanner())

    if (solved):
        ss.simplifySolution()
        path = ss.getSolutionPath() 
        print(len(path.getStates()))
        for state in path.getStates():
            print(state)

if __name__ == "__main__":
    plan()

