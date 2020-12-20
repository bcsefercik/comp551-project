"""
import numpy as np
import matplotlib.pyplot as plt

background = [0.97,0.95,0.98,0.98,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.98,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.98,0.99,0.99]
arm = [0.59, 0.81, 0.96, 0.96, 0.97, 0.98, 0.91, 0.96, 0.91, 0.94, 0.94, 0.97, 0.95, 0.97, 0.59, 0.97, 0.99, 0.99, 0.33, 0.99, 0.96]

data = [background,arm]
X = np.arange(len(background))
print(X)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, data[0], color = 'b', width = 0.50)
ax.bar(X + 0.25, data[1], color = 'g', width = 0.50)

"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)

labels     = ['1',  '2',  '3',   '4', '5',  '6',  '7',  '8',   '9',  '10', '11', '12', '13', '14', '15' , '16',  '17', '18', '19', '20', '21']
background = [0.97, 0.95, 0.98, 0.98, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.98, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.98, 0.99,0.99]
arm        = [0.59, 0.81, 0.96, 0.96, 0.97, 0.98, 0.91, 0.96, 0.91, 0.94, 0.94, 0.97, 0.95, 0.97, 0.59, 0.97, 0.99, 0.99, 0.33, 0.99, 0.96]

print(len(labels))
print(len(background))
print(len(arm))

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, background, width, label='Background')
rects2 = ax.bar(x + width/2, arm, width, label='Arm')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('$mAP_{50}$',fontsize=15)
ax.set_title('Validation set mAP results')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


#autolabel(rects1)
#autolabel(rects2)

fig.tight_layout()

plt.show()