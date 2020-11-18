import torch
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
import os
from models.modules_causal_vel import *
from data.AL_sampler import RandomPytorchSampler
from data.datasets import *
from data.dataset_utils import *
import argparse
from torch.utils.data import DataLoader
from utils.functions import *


class AbstractPlotScenario(object):
    def __init__(self, im_size, duration, cube_len, gt_hsl, pred_hsl):
        """
        Hue-Saturation-Lightness (HSL) functions, given as hsl(hue, saturation%, lightness%) where hue is the color given as an angle between 0 and 360 (red=0, green=120, blue=240), saturation is a value between 0% and 100% (gray=0%, full color=100%), and lightness is a value between 0% and 100% (black=0%, normal=50%, white=100%). For example, hsl(0,100%,50%) is pure red.

        Args:
            im_size ([type]): [description]
            duration ([type]): [description]
            cube_len ([type]): [description]
            gt_hsl ([type]): [description]
            pred_hsl ([type]): [description]
        """
        # 255, 255, 255 is white
        self.im_size = im_size
        self.duration = duration
        self.cube_len = cube_len
        self.pred_hsl = pred_hsl
        self.gt_hsl = gt_hsl

    def draw_background(self, *args):
        raise NotImplementedError

    def draw_trajectory(self, *args):
        raise NotImplementedError

    def draw_cube_stack(self, folder, trajectory, scale, save, batch_ind=-1, suffix='', draw_second_cube=True, title=None, **kwargs):
        """[summary]

        Args:
            folder ([type]): [description]
            trajectory ([type]): gt first, pred second
            scale ([type], optional): [description]. Defaults to None.
            suffix (str, optional): [description]. Defaults to ''.
            save (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        images = []
        for i in range(trajectory.shape[1]):
            print(i, trajectory[0][i]*scale, trajectory[1][i]*scale)
            fig = plt.figure(figsize=(8, 8))
            pos_1 = trajectory[0][i]*scale
            im, draw, start, _ = self.draw_background(**kwargs)
            self.draw_trajectory(draw, start, pos_1,
                                 'hsl(%d, %d%%, %d%%)' % self.gt_hsl, **kwargs)
            if draw_second_cube:
                pos_2 = trajectory[1][i]*scale
                self.draw_trajectory(draw, start, pos_2,
                                     'hsl(%d, %d%%, %d%%)' % self.pred_hsl, **kwargs)
            if not title:
                fig.suptitle('Blue ground truth, red prediction.')
            else:
                fig.suptitle(title)
            plt.xlim([0, self.im_size[0]])
            plt.ylim([self.im_size[1], 0])
            plt.autoscale(False)
            plt.imshow(im)
            fig.savefig(os.path.join(folder, 'temp.png'))
            images.append(Image.open(os.path.join(folder, 'temp.png')))
        if save:
            images[0].save(os.path.join(folder, '_'.join([suffix, str(batch_ind), 'cube_stack.gif'])),
                           save_all=True, append_images=images[1:], optimize=False, duration=self.duration, loop=0)
        return images


class Plot_FrictionSliding(AbstractPlotScenario):
    def __init__(self, im_size, duration=400, cube_len=30, gt_hsl=(180, 100, 50), pred_hsl=(0, 100, 50)):
        AbstractPlotScenario.__init__(
            self, im_size, duration, cube_len, gt_hsl, pred_hsl)
        # self.slope_len = slope_len

    def draw_background(self, theta=None, line_width=5):
        #theta in degree
        #     theta = theta * np.pi/180
        start_x = self.im_size[0]//8
        start_y = self.im_size[1]//8
        # Want the slope to in the middle 3/4
        slope_len = int((self.im_size[1]*0.75)/np.sin(theta))
        end_x = start_x+int(slope_len*np.cos(theta))
        end_y = 7*self.im_size[1]//8
        im = Image.new('RGB', self.im_size, (255, 255, 255))
        draw = ImageDraw.Draw(im)
        draw.line([(start_x, start_y), (end_x, end_y)],
                  width=line_width, fill=(0, 0, 0))
        return im, draw, (start_x, start_y), slope_len

    def draw_trajectory(self, draw, start, pos, fill, theta=None):
        #     theta = theta * np.pi/180
        cx = start[0]+pos*np.cos(theta)
        cy = start[1]+pos*np.sin(theta)
        cx += np.sin(theta)*self.cube_len/2
        cy -= np.cos(theta)*self.cube_len/2
        x_ll = cx-np.cos(45 * np.pi/180 - theta)*self.cube_len/np.sqrt(2)
        y_ll = cy+np.sin(45 * np.pi/180 - theta)*self.cube_len/np.sqrt(2)
        x_lr = cx+np.sin(45 * np.pi/180 - theta)*self.cube_len/np.sqrt(2)
        y_lr = cy+np.cos(45 * np.pi/180 - theta)*self.cube_len/np.sqrt(2)

        x_ul = cx+np.cos(45 * np.pi/180 - theta)*self.cube_len/np.sqrt(2)
        y_ul = cy-np.sin(45 * np.pi/180 - theta)*self.cube_len/np.sqrt(2)
        x_ur = cx-np.sin(45 * np.pi/180 - theta)*self.cube_len/np.sqrt(2)
        y_ur = cy-np.cos(45 * np.pi/180 - theta)*self.cube_len/np.sqrt(2)
        draw.polygon([(x_ul, y_ul), (x_lr, y_lr),
                      (x_ll, y_ll), (x_ur, y_ur)], fill=fill)


class Plot_FrictionlessSHO(AbstractPlotScenario):
    def __init__(self, im_size, duration=400, cube_len=30, gt_hsl=(180, 100, 50), pred_hsl=(0, 100, 50)):
        AbstractPlotScenario.__init__(
            self, im_size, duration, cube_len, gt_hsl, pred_hsl)

    def draw_background(self, line_width=5):
        start_x = self.im_size[0]//2
        start_y = 7*(self.im_size[1]//8)
        im = Image.new('RGB', self.im_size, (255, 255, 255))
        draw = ImageDraw.Draw(im)
        draw.line([(0, start_y), (self.im_size[0], start_y)],
                  width=line_width, fill=(0, 0, 0))
        return im, draw, (start_x, start_y), None

    def draw_trajectory(self, draw, start, pos, fill):
        cx = start[0]+pos
        cy = start[1]

        x_ll = cx-self.cube_len/2
        y_ll = cy
        x_lr = cx+self.cube_len/2
        y_lr = cy

        x_ul = cx-self.cube_len/2
        y_ul = cy-self.cube_len
        x_ur = cx+self.cube_len/2
        y_ur = cy-self.cube_len
        draw.polygon([(x_ul, y_ul), (x_ur, y_ur),
                      (x_lr, y_lr), (x_ll, y_ll)], fill=fill)


class Plot_AirFall(AbstractPlotScenario):
    def __init__(self, im_size, duration=400, cube_len=30, gt_hsl=(180, 100, 50), pred_hsl=(0, 100, 50)):
        # (180, 100, 50) is color aqua.
        AbstractPlotScenario.__init__(
            self, im_size, duration, cube_len, gt_hsl, pred_hsl)

    def draw_background(self):
        start_x = self.im_size[0]//2
        start_y = 50
        im = Image.new('RGB', self.im_size, (255, 255, 255))
        draw = ImageDraw.Draw(im)
        return im, draw, (start_x, start_y), None

    def draw_trajectory(self, draw, start, pos, fill):
        cx = start[0]
        cy = start[1]+pos

        x_ll = cx-self.cube_len/2
        y_ll = cy+self.cube_len/2
        x_lr = cx+self.cube_len/2
        y_lr = cy+self.cube_len/2

        x_ul = cx-self.cube_len/2
        y_ul = cy-self.cube_len/2
        x_ur = cx+self.cube_len/2
        y_ur = cy-self.cube_len/2
        draw.polygon([(x_ul, y_ul), (x_ur, y_ur),
                      (x_lr, y_lr), (x_ll, y_ll)], fill=fill)


class Plot_Connections(AbstractPlotScenario):
    def __init__(self, im_size, duration=400, cube_len=30, gt_hsl=(55, 100, 50), pred_hsl=(180, 100, 100), center_xs=(100, 300)):
        """[summary]

        Args:
            im_size ([type]): [description]
            duration (int, optional): [description]. Defaults to 400.
            cube_len (int, optional): [description]. Defaults to 30.
            gt_hsl (tuple, optional): Here the gt's are the circles indicating input variables. Defaults to (180, 100, 50).
            pred_hsl (tuple, optional): Here the pred's are the connection lines. Defaults to (0, 100, 50).
            center_xs (tuple, optional): [description]. Defaults to (100, 300).
        """
        AbstractPlotScenario.__init__(
            self, im_size, duration, cube_len, gt_hsl, pred_hsl)
        self.center_xs = center_xs

    def read_A(self, path, epoch):
        path = os.path.join(path, str(epoch)+'_decoder.pt')
        A = torch.load(path)[1].softmax(-1).squeeze()
        self.num_nodes = int(np.sqrt(A.size(0)))
        A = A.view(self.num_nodes, self.num_nodes, -1).cpu().detach().numpy()
        return A

    def draw_background(self, texts):
        r = 0.4*self.im_size[1]/(2+self.num_nodes)
        im = Image.new('RGB', self.im_size, (255, 255, 255))
        draw = ImageDraw.Draw(im)
        self.center_ys = np.linspace(
            0, self.im_size[1], self.num_nodes+1, endpoint=False)[1:]
        for x in self.center_xs:
            for i in range(len(self.center_ys)):
                y = self.center_ys[i]
                draw.ellipse((x-r, y-r, x+r, y+r),
                             fill='hsl(%d, %d%%, %d%%)' % self.gt_hsl)
                draw.text((x-r, y-r), texts[i], fill=(0, 0, 0))
        return im, draw, None, None

    def draw_trajectory(self, draw, A, line_width):
        edge = A.argmax(-1)
        for i in range(len(A)):
            for j in range(len(A[0])):
                if edge[i][j] == 1:
                    # print(A[i][j][1])
                    color = 'hsl(%d, %d%%, %d%%)' % (
                        self.pred_hsl[0], (1-A[i][j][1])*self.pred_hsl[1], (1-A[i][j][1])*self.pred_hsl[1])
                    draw.line([(self.center_xs[0], self.center_ys[j]), (self.center_xs[1], self.center_ys[i])],
                              width=line_width, fill=color)

    def draw_connection_gif(self, folder, epochs, save, suffix='', line_width=3, texts=["shape", "color", "friction", "theta", "mass", "x_0", "vel", "pos"]):
        images = []
        # try:
        for i in epochs:
            A = self.read_A(folder, int(i))
            im, draw = self.draw_background(texts)
            self.draw_trajectory(draw, A, line_width)
            images.append(im)
        # except:
        #     pass
        if save:
            images[0].save('_'.join([folder, suffix, 'connection.gif']),
                           save_all=True, append_images=images[1:], optimize=False, duration=self.duration, loop=0)
        return images
