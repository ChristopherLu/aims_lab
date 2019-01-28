#
# IPS Online Submission Script
# ( Cyber Physical System Group )
# Jan. 2019
# Peijun Zhao, Chris Xiaoxuan Lu
#

import requests
import statistics

errors = list()
solution = open('./real_data/test/location.txt','r')

i = 1
for line in solution:
    data = line.strip().split(',')
    x = float(data[0])
    y = float(data[1])
    test_data = {'x':x,'y':y}
    requrl = 'http://47.89.177.245:5000/{}'.format(i)
    r = requests.get(requrl, test_data)
    errors.append(float(r.text))
    print("Point {} tested, prediction error is {:.2f} m".format(i, float(r.text)))
    i+=1

print('==========')
print('Average Prediction Error: {:.2f} m'.format(statistics.mean(errors)))