import os
import pydrake.all

def FindResource(filename):
    return os.path.join(os.path.dirname(__file__), filename)

def AddPackagePaths(parser):
    parser.package_map().PopulateFromFolder(FindResource(""))
    parser.package_map().Add(
        "manipulation_station",
        os.path.join(pydrake.common.GetDrakePath(),
                     "examples/manipulation_station/models"))




