import pydrake.all
from scenarios import AddPanda, AddPandaHand

def MakePandaStation(time_step = 0.002):
    builder = pydrake.systems.framework.DiagramBuilder()


