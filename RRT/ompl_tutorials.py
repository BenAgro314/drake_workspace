from ompl import base as ob
from ompl import geometric as og


param = 0.6

class TestClass:

    def __init__(self, param1):
        self.param1 = param1

    def isStateValid(self, state):
        print(state.rotation().x)
        return True

def isStateValid(state):
    global param
    return state[0] < param

def plan():

    test = TestClass(0.6)


    dim = 2
    space = ob.SE3StateSpace()
    bounds = ob.RealVectorBounds(3)
    bounds.setLow(-10)
    bounds.setHigh(10)
    space.setBounds(bounds)

    ss = og.SimpleSetup(space)
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(test.isStateValid))
    state = ob.State(space)

    start = ob.State(space)
    start[0] = 0.5
    start[1] = 0
    start[2] = 0.5
    start[3] = 8.7e-18
    start[4] = 1 
    start[5] = 6.16e-17
    start[6] = 6.16e-17
    print(start)

    goal = ob.State(space)
    goal[0] = 0.6
    goal[1] = 0
    goal[2] = 0.2694
    goal[3] = -6.12e-17
    goal[4] = 1
    goal[5] = -6.12e-17
    goal[6] = 1.61e-15
    print(goal)

    ss.setStartAndGoalStates(start, goal)

    solved = ss.solve()

    if (solved):
        ss.simplifySolution()
        path = ss.getSolutionPath() 
        print(len(path.getStates()))
        for state in path.getStates():
            print(state)

if __name__ == "__main__":
    plan()

