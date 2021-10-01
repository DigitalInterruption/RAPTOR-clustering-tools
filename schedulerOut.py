import yaml
import os

outDict = {'done':True}
with open('control/schedulerClean.yaml', 'w') as file:
    dump = yaml.dump(outDict, file)
