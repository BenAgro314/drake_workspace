from ompl import base as ob
from ompl import geometric as og


param = 0.6

class TestClass:

    def __init__(self, param1):
        self.param1 = param1

    def isStateValid(self, state):
        return state[0] < self.param1

def isStateValid(state):
    global param
    return state[0] < param

def plan():

    test = TestClass(0.6)


    dim = 2
    space = ob.RealVectorStateSpace(dim)
    bounds = ob.RealVectorBounds(dim)
    joint_limits = [(0, 1), (0,2), (-1, 2),(-1, 2),(-1, 2),(-1, 2),(-1, 2)]
    for i in range(dim):
        bounds.setLow(i, joint_limits[i][0])
        bounds.setHigh(i, joint_limits[i][1])
    space.setBounds(bounds)

    ss = og.SimpleSetup(space)
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(test.isStateValid))
    state = ob.State(space)

    start = ob.State(space)
    start[0] = 0.1
    start[1] = 0.1 

    goal = ob.State(space)
    goal[0] = 0.3
    goal[1] = 0.4

    ss.setStartAndGoalStates(start, goal)

    solved = ss.solve(100)

    if (solved):
        ss.simplifySolution()
        path = ss.getSolutionPath() 
        print(len(path.getStates()))
        for state in path.getStates():
            print(state[0], state[1])

if __name__ == "__main__":
    plan()

