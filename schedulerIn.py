import yaml
import time

outDict = {'done':False}
with open('control/schedulerClean.yaml', 'w') as file:
    dump = yaml.dump(outDict, file)

while True:
    time.sleep(5)
    with open('control/scheduler.yaml', 'r') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    if params['done']:
        print('do', params['done'])
        break
    else:
        print('waiting')
