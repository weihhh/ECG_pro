import matplotlib.pyplot as plt
import moviepy.editor as mpy
from matplotlib import cm, colors
from moviepy.video.io.bindings import mplfig_to_npimage
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as manimation

datax=[-94.3900 ,-64.6400 ,-50.3000 ,-37.7200 ,-10.1600 ]
datay=[1.9500 ,7.5600, 11.0100 ,14.3100 ,15.7400]
dataz=[190.8700 ,229.4000, 243.5700, 254.3100 ,267.6900]


class Charts_2d(object):
    def __init__(self, all_x=None, all_y=None, lim_x=None, lim_y=None,dots_list=None):
        self.fig_2d = plt.figure(1)
        self.ax1=self.fig_2d.add_subplot(111)
        self.ax1.grid(True,color='k')#打开网格
        if all_x is not None and all_y is not None:
            self.obj_x = all_x
            self.obj_y = all_y
        if lim_x is not None:
            self.ax1.set_xlim(lim_x)#lim_x最好传入[0,120]这样的范围，否则反向
        if lim_y is not None:
            self.ax1.set_ylim(lim_y)

        self.dots_list=dots_list
        self.fps = 100
        self.dots_plottimes=0

    def draw_static(self):
        
        self.ax1.set_xlabel('x')
        self.ax1.set_ylabel('y')
        
        self.ax1.scatter(self.obj_x,self.obj_y,c='r',marker='o',label='o_data')
        self.ax1.plot(self.obj_x, self.obj_y)
        
        plt.show()

    def draw_dynamic(self, is_point=True, is_line=True, is_color_map=True):
        color_map = plt.get_cmap('jet')
        color_norm = colors.Normalize(vmin=min(self.obj_y), vmax=max(self.obj_y))
        scalar_map = cm.ScalarMappable(norm=color_norm, cmap=color_map)
        count = 0
        while count < len(self.obj_y):
            if is_point:
                if is_color_map:
                    self.ax1.scatter(self.obj_x[count], self.obj_y[count],
                        c=scalar_map.to_rgba(self.obj_y[count]), marker='.', s=30, label='')
                else:
                    self.ax1.scatter(self.obj_x[count], self.obj_y[count],
                        c='r', marker='.', s=50, label='')
            # in same fig, every call will change color, two point make one line.
            # ax.plot_wireframe won't
            if is_line:
                self.ax1.plot(self.obj_x[count:count + 2], self.obj_y[count:count + 2])
            plt.pause(.00001)#这里要有延迟，否则会不显示绘图过程
            self.writer.grab_frame()#注意这里的抓图，很重要
            # plt.show()
            count += 1
    def save_mp4(self):
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Movie Test', artist='Matplotlib',
        comment='Movie support!')
        self.writer = FFMpegWriter(fps=self.fps, metadata=metadata)#调节fps可以改变显示速度
        with self.writer.saving(self.fig_2d, "dat4_grid.mp4", 500):#500为dpi
            self.draw_dynamic()

    def plot_dot_list(self,t):
        dot_add=10
        self.dots_plottimes+=1

        for dots,color in zip(self.dots_list,['r','b','k','g','y']):
            # self.ax1.scatter(dots[0][t], dots[1][t], marker='.',linewidth=3, s=30, label='',color=color)
            self.ax1.plot(dots[0][t:t + dot_add], dots[1][t:t + dot_add],linewidth=3,color=color)


    # change color direction by Normalize and to_rgba's method
    def make_frame_dynamic(self, t):#返回t时刻frame，t*帧数等于第几张图
        t = int(t * self.fps)
        # print(t)
        dot_add=10
        if not self.dots_list:
            color_map = plt.get_cmap('jet')
            color_norm = colors.Normalize(vmin=min(self.obj_y), vmax=max(self.obj_y))
            scalar_map = cm.ScalarMappable(norm=color_norm, cmap=color_map)
            self.ax1.scatter(self.obj_x[t], self.obj_y[t],
                c=scalar_map.to_rgba(self.obj_y[t]), marker='.', s=30, label='')
            self.ax1.plot(self.obj_x[t:t + dot_add], self.obj_y[t:t + dot_add])
            print(t, end=' ')
        else:
            self.plot_dot_list(t)

        return mplfig_to_npimage(self.fig_2d)#产生frame

    # gif file size is big than video
    #一帧一个点默认
    def write_gif(self, path, fps=10):
        speedx=10
        animation = mpy.VideoClip(self.make_frame_dynamic, duration=10)
        self.fps = fps
        # animation.write_gif(path, fps=self.fps)
        animation.speedx(speedx).to_gif(path,fps=fps)#durationXfps=speed X dots,dots表示打点个数
        print(self.dots_plottimes)
    
    def write_video_2d(self, path, fps=10):
        animation = mpy.VideoClip(self.make_frame_dynamic, duration=5)
        self.fps = fps
        animation.write_videofile(path, fps=self.fps)

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




# one_charts=Charts_2d(datax,datay)
# one_charts.draw_static()
