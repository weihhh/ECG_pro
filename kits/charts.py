#!/usr/bin/env python
# -*- coding: utf-8 -*-
# import cv2
import matplotlib.pyplot as plt
import moviepy.editor as mpy
from matplotlib import cm, colors
from moviepy.video.io.bindings import mplfig_to_npimage
from mpl_toolkits.mplot3d import Axes3D

class Charts(object):
    def __init__(self, all_x=None, all_y=None, all_z=None, lim_x=None, lim_y=None,
                 lim_z=None):
        self.fig_3d = plt.figure(1)
        self.ax_3d = Axes3D(self.fig_3d)
        if all_x is not None and all_y is not None:
            self.obj_x = all_x
            self.obj_y = all_y
        if all_z is not None:
            self.obj_z = all_z
        if lim_x is not None:
            self.ax_3d.set_xlim(lim_x)
        if lim_y is not None:
            self.ax_3d.set_ylim(lim_y)
        if lim_z is not None:
            self.ax_3d.set_zlim(lim_z)
        self.fps = 10

    def draw_static(self):
        
        self.ax_3d.set_xlabel('x')
        self.ax_3d.set_ylabel('y')
        self.ax_3d.set_zlabel('z')
        #散点加上线
        self.ax_3d.scatter(self.obj_x, self.obj_y, self.obj_z, c='r', marker='.', s=30, label='')
        self.ax_3d.plot(self.obj_x, self.obj_y, self.obj_z)
        plt.show()

    def draw_dynamic(self, is_point=True, is_line=True, is_color_map=True):
        color_map = plt.get_cmap('jet')
        color_norm = colors.Normalize(vmin=min(self.obj_z), vmax=max(self.obj_z))
        scalar_map = cm.ScalarMappable(norm=color_norm, cmap=color_map)
        count = 0
        while count < len(self.obj_z):
            self.ax_3d.view_init(30, count * 2)
            if is_point:
                if is_color_map:
                    self.ax_3d.scatter(self.obj_x[count], self.obj_y[count], self.obj_z[count],
                        c=scalar_map.to_rgba(self.obj_z[count]), marker='.', s=30, label='')
                else:
                    self.ax_3d.scatter(self.obj_x[count], self.obj_y[count], self.obj_z[count],
                        c='r', marker='.', s=50, label='')
            # in same fig, every call will change color, two point make one line.
            # ax.plot_wireframe won't
            if is_line:
                self.ax_3d.plot(self.obj_x[count:count + 2], self.obj_y[count:count + 2],
                    self.obj_z[count:count + 2])
            plt.pause(.00001)
            # plt.show()
            count += 1

    # change color direction by Normalize and to_rgba's method
    def make_frame_dynamic(self, t):
        t = int(t * self.fps)
        self.ax_3d.view_init(30, t * 2)  # angle
        color_map = plt.get_cmap('jet')
        color_norm = colors.Normalize(vmin=min(self.obj_z), vmax=max(self.obj_z))
        scalar_map = cm.ScalarMappable(norm=color_norm, cmap=color_map)
        self.ax_3d.scatter(self.obj_x[t], self.obj_y[t], self.obj_z[t],
            c=scalar_map.to_rgba(self.obj_z[t]), marker='.', s=30, label='')
        self.ax_3d.plot(self.obj_x[t:t + 2], self.obj_y[t:t + 2],
            self.obj_z[t:t + 2])
        print(t, end=' ')
        return mplfig_to_npimage(self.fig_3d)

    # gif file size is big than video
    def write_gif(self, path, fps=10):
        animation = mpy.VideoClip(self.make_frame_dynamic, duration=10)
        self.fps = fps
        animation.write_gif(path, fps=self.fps)

    def write_video(self, path, fps=10):
        self.fps = fps
        writer = cv2.VideoWriter()
        first_image = self.make_frame_dynamic(0)
        print(first_image.shape)
        writer.open(path, cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), self.fps,
            (first_image.shape[1], first_image.shape[0]))
        writer.write(first_image)
        count = 0
        while count < len(self.obj_z) - 1:
            count += 1
            writer.write(self.make_frame_dynamic(count / self.fps))


datax=[-94.3900 ,-64.6400 ,-50.3000 ,-37.7200 ,-10.1600 ]
datay=[1.9500 ,7.5600, 11.0100 ,14.3100 ,15.7400]
dataz=[190.8700 ,229.4000, 243.5700, 254.3100 ,267.6900]

# one_charts=Charts(datax,datay,dataz)
# one_charts.draw_dynamic()