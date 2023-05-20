"""visualise and test the results of the model"""
import glob
import os
import random
import sys
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from p2mUtils.viz import plotCubicSpline, plot_distance_field
from dfUtils.cubicCurvesUtil import *
import math
import random
from loss_functions import *
from pathlib import Path


class ModelResults:

    def __init__(self, outputs):
        self.outputs = outputs
        self.pkl_dict = self.load_pkl_files()

    def select_keys(self, n=10, limit=50):
        return random.sample(list(self.pkl_dict.keys())[:limit], n)


    def process_data(self, key):
        path = Path(self.pkl_dict[key][0][0])
        filename = path.name
        points = self.pkl_dict[key][1]
        n = points.shape[0]
        n /= 3
        points = torch.reshape(points, (int(n), 6)).unsqueeze(0)
        path = self.pkl_dict[key][0][0]
        data = torch.load(path)
        df_path = path.replace("spoints", "distance_field")
        df = torch.load(df_path)
        return path, filename, points, data, df

    def calculate_distance_field_and_loss(self, points, data, df):
        grid_size = 224
        ctrl_points = convert_to_cubic_control_points(points).to(th.float64)
        source_points = create_grid_points(grid_size, 0, 250, 0, 250).to(th.float64)
        source_points = source_points.to('cuda')
        distance_field = distance_to_curves(source_points, ctrl_points, grid_size).view(grid_size, grid_size)
        distance_field = th.flip(distance_field, (1,))
        distance_field = distance_field / torch.max(distance_field)
        dfloss = distance_field_loss(distance_field, df)
        return distance_field, dfloss, ctrl_points

    def load_pkl_files(self):
        pkl_files = glob.glob(self.outputs + "/*.pkl")
        pkl_files.sort()
        pkl_dict = {}
        for pkl_file in pkl_files:
            pkl_dict[os.path.basename(pkl_file)] = pickle.load(open(pkl_file, 'rb'), encoding='bytes')
        for key in pkl_dict.keys():
            pkl_dict[key][1] = torch.from_numpy(pkl_dict[key][1])
        return pkl_dict

    def plot_data(self, key, path, filename, points, data, df):
        fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(6, 24))
        plot_distance_field(df, 1, title=f"GT Distance Field for {key}", ax=axs[0])
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[4].set_xticks([])
        axs[4].set_yticks([])

        y = data['y'].permute(1, 2, 0)
        pred_df,loss,ctrl_points_pred= self.calculate_distance_field_and_loss(points, data, df)
        points = data['x'].unsqueeze(0)
        ctrl_points = convert_to_cubic_control_points(points)
        plotCubicSpline(ctrl_points, title=f"gt {key}", ax=axs[2])

        axs[1].imshow(y)
        plotCubicSpline(ctrl_points_pred, title=f"prediction {key}", ax=axs[4])
        plot_distance_field(pred_df, 1, title=f"Pred Distance Field for {key}", ax=axs[3])
        # Split the title into two lines and adjust the vertical position
        axs[1].set_title(key)
        fig.suptitle(f"dfloss={loss}\nfor {key}, {filename}", y=0.95)
        # Increase the top margin to make more space for the title
        plt.subplots_adjust(top=0.9)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.show()


if __name__ == "__main__":
    outputs = r"D:\pyG\temp\RES\05-04_07-18-39\epoch_11\outputs"
    model_results = ModelResults(outputs)
    keys = model_results.select_keys(n=20, limit=100)

    for key in keys:
        path, filename, points, data, df = model_results.process_data(key)
        model_results.plot_data(key, path, filename, points, data, df)
        plt.tight_layout()
        plt.show()

