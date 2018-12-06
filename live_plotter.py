__author__ = 'LobaAjisafe'
# Idea taken from https://github.com/engineersportal/pylive

import matplotlib.pyplot as plt
import numpy as np

# use ggplot style for more sophisticated visuals
plt.style.use('ggplot')


def live_plotter(x_vec, y1_data, line1, line2=None, y2_data=None, identifier='', pause_time=0.1):
    if line2 is None:
        if line1 == []:
            # this is the call to matplotlib that allows dynamic plotting
            plt.ion()
            fig = plt.figure(figsize=(13, 6))
            ax = fig.add_subplot(111)
            # create a variable for the line so we can later update it
            line1, = ax.plot(x_vec, y1_data, '-o', alpha=0.8)
            # update plot label/title
            plt.ylabel('Y Label')
            plt.title('Title: {}'.format(identifier))
            plt.show()

        # after the figure, axis, and line are created, we only need to update the y-data
        line1.set_ydata(y1_data)
        # adjust limits if new data goes beyond bounds
        if np.min(y1_data) <= line1.axes.get_ylim()[0] or np.max(y1_data) >= line1.axes.get_ylim()[1]:
            plt.ylim([np.min(y1_data)-np.std(y1_data), np.max(y1_data)+np.std(y1_data)])
        # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
        plt.pause(pause_time)

        # return line so we can update it again in the next iteration
        return line1
    else:
        if line1 == [] and line2 == []:
            # this is the call to matplotlib that allows dynamic plotting
            plt.ion()
            fig = plt.figure(figsize=(13, 6))
            ax = fig.add_subplot(111)
            ax2 = ax.twinx()
            # create a vari able for the line so we can later update it
            line1 = ax.plot(x_vec, y1_data, '-o', alpha=0.8)
            line2 = ax2.plot(x_vec, y2_data, '-o', alpha=0.8, c='g')
            # update plot label/title
            plt.ylabel('Y Label')
            plt.title('Title: {}'.format(identifier))
            plt.show()
            line1 = line1[0]
            line2 = line2[0]

        # after the figure, axis, and line are created, we only need to update the y-data
        line1.set_ydata(y1_data)
        line2.set_ydata(y2_data)
        # adjust limits if new data goes beyond bounds
        if np.min(y1_data) <= line1.axes.get_ylim()[0] or np.max(y1_data) >= line1.axes.get_ylim()[1]:
            plt.ylim([np.min(y1_data)-np.std(y1_data), np.max(y1_data)+np.std(y1_data)])
            plt.ylim([np.min(y2_data)-np.std(y2_data), np.max(y2_data)+np.std(y2_data)])
        # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
        plt.pause(pause_time)

        # return line so we can update it again in the next iteration
        return line1, line2
