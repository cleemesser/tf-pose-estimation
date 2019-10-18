# can use with
# %matplotlib tk

import time
import matplotlib 

import matplotlib.pyplot as plt
import numpy as np 

xy = np.load('xy.npy') # shape=(2,18)
xys = np.load('xys.npy')

# next add connections between relevant body parts
class skeleton_plot:
    def __init__(self, xys):
        """xys = shape (18,2,389) -> (part#, xydim, timestep)"""
        self.xys = xys
        self.t = 0
        xy = xys[:,:,self.t]
        self.N = len(xys)
        plt.ion  # turn on interactive plot
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        ax = self.ax
        self.lines = ax.plot(xy[:,0],xy[:,1], 'o')
                
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0,1.0)
        # ax.set_ylim(ax.get_ylim()[::-1])
        ax.invert_yaxis()
        ax.set_aspect('equal')

        self.t += 1
        
    def update(self):
        line0, = self.lines
        xy = self.xys[:,:,self.t]
        line0.set_xdata(xy[:,0])
        line0.set_ydata(xy[:,1] )
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.t += 1

    def loop_all(self, times, delay=0.01):
        """times is an iterable"""
        for tt in times:
            self.t = tt
            self.update()
            time.sleep(delay)


if __name__ == '__main__':
    p = skeleton_plot(xys)
    p.loop_all(times=range(p.N), delay=0.3)