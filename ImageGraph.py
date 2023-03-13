"""class for the graph represetation of an image"""
import os

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
import patchify
import cv2
import plotly.graph_objects as go
import plotly.offline as py
from sklearn.neighbors import KNeighborsClassifier

class ImageGraph:
    """Points are represented as Nodes with edges between them."""
    def __init__(self, imagePath):
        """take an ordered list of points and create a graph representation of the shape"""
        self.imagePath = imagePath
        self.image = None
        self.nodes = []
        self.edges=[]
        self.x=[]
        self.loadImage()
        self.createPatches()
        self.createNodes()
        self.findKnearestNeighbours(2)
        self.createEdges()
        self.getGraph()

    def loadImage(self):

        #check image path exists
        if os.path.exists(self.imagePath):
            print("image path exists")
            self.image = cv2.imread(self.imagePath)
            self.image = cv2.resize(self.image, (128, 128))
        else:
            print("image path does not exist")
            # don't create class
            return

    def createPatches(self):
        """For an image with size of H × W × 3, we divided it into N patches"""
        maxPatchSize = 32
        self.patches = patchify.patchify(self.image, (maxPatchSize, maxPatchSize, 3), step=maxPatchSize)
        self.nPatches = self.patches.shape[0] * self.patches.shape[1]

    def patchToFeatureVector(self, patch):
        """Convert a patch to a feature vector"""
        return patch.flatten()
    def createNodes(self):
        """create the nodes from the patches"""
        for i in range(self.patches.shape[0]):
            for j in range(self.patches.shape[1]):
                self.nodes.append(self.patchToFeatureVector(self.patches[i, j, :, :, :]))
        self.x = torch.tensor(np.array(self.nodes), dtype=torch.float)

    def findKnearestNeighbours(self, k):
        """find the k nearest neighbours for each patch"""
        self.indices = []
        self.distances = []
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(self.x, self.x)
        self.indices = neigh.kneighbors(self.x, return_distance=False)
        self.distances = neigh.kneighbors(self.x, return_distance=True)[0]

    def createEdges(self):
        """create the edges from the indices"""
        #swap the indices columns so we build and edge from Vj to Vi
        self.edge_index = np.fliplr(self.indices)

    def getGraph(self):
        """return the graph"""
        self.data=Data(x=self.x, edge_index=torch.tensor(self.edge_index.copy(), dtype=torch.long).t().contiguous())
        return self.data

    def visualiseGraph(self):
        """visualise the graph"""
        G = to_networkx(self.data)
        pos=nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, font_weight='bold')


        #nx.draw(G, with_labels=True, font_weight='bold')
        plt.show()

    def displayPatches(self):
        """display the patches with their edge index in the same plot"""
        fig, axs = plt.subplots(self.patches.shape[0], self.patches.shape[1])
        for i in range(self.patches.shape[0]):
            for j in range(self.patches.shape[1]):
                axs[i, j].imshow(self.patches[i, j][0])
                axs[i, j].set_title(str(self.edge_index[i, j]))
        plt.show()






if __name__ == '__main__':
    image = ImageGraph("data/Capture.JPG")
    data=image.getGraph()
    #get number of node features
    num_node_features = data.x.shape[1]
    #from models.myGNN import GCN
    #model=GCN(num_node_features, 8)
    #gnn=model(data)



