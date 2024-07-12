from pylab import *

with open('test3.txt') as f:
    lines = [line.rstrip('\n') for line in f]

tip_x = []
tip_y = []

heel_x = []
heel_y = []

foot_x = []
foot_y = []

knee_x = []
knee_y = []

for i in lines:
    knee_x.append(float(i.split()[4]))
    knee_y.append(float(i.split()[5]))

    foot_x.append(float(i.split()[6]))
    foot_y.append(float(i.split()[7]))

    heel_x.append(float(i.split()[8]))
    heel_y.append(float(i.split()[9]))

    tip_x.append(float(i.split()[10]))
    tip_y.append(float(i.split()[11]))

scatter(knee_y,knee_x,s=100, marker='.')
title("knee")
show()
scatter(heel_y,heel_x,s=100, marker='.')
title("heel")
show()
scatter(tip_y,tip_x,s=100, marker='.')
title("tip")
show()
scatter(foot_y,foot_x,s=100, marker='.')
title("foot")
show()
