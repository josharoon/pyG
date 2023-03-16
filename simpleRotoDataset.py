"""simpleRotoDataset.py - A simple dataset for testing and training rotoscoping models."""
import math

import torch
from pathlib import Path
from torch_geometric.data import Data, Dataset,download_url
from torchvision.io import read_image
from nkShapeGraph import ShapeGraph,point2D
#from ImageGraph import ImageGraph
import json

class SimpleRotoDataset(Dataset):
    """A simple dataset for testing and training rotoscoping models."""
    def __init__(self, root, transform=None, pre_transform=None,pre_filter=None):
        super().__init__(root, transform, pre_transform,pre_filter)
        self.root=root




    def loadLabelsJson(self):
        """load the json file containing the labels"""
        with open(self.labels) as f:
            data = json.loads(f.read())
        return data
    def Shape2Point2D(self,shape):
        """convert an json point to a point2D object e.g {'center': { 362, 240, 1 }, 'leftTangent': { 9, 4, 1 }, 'rightTangent': { 9, -2, 1 }"""
        #if string change property names to double quotes and convert back to dict
        #comvert shape string to dict
        if isinstance(shape,str):
            shape=shape.replace("'",'"')
            shape=json.loads(shape)
        #remove { from values convert to list of floats e.g. { 362, 240, 1 } to [362.0, 240.0, 1.0]
        shape['center']=shape['center'].replace('{','').replace('}','').split(',')
        shape['center']=[float(x) for x in shape['center']]
        shape['leftTangent']=shape['leftTangent'].replace('{','').replace('}','').split(',')
        shape['leftTangent']=[float(x) for x in shape['leftTangent']]
        shape['rightTangent']=shape['rightTangent'].replace('{','').replace('}','').split(',')
        shape['rightTangent']=[float(x) for x in shape['rightTangent']]
        #convert to point2D object
        return point2D(shape['center'],shape['leftTangent'],shape['rightTangent'])
    def getPoints2DList(self, pointList):
        """convert a list of json points to a list of point2D objects"""
        return [self.Shape2Point2D(shape) for shape in pointList]

    def process(self):
        """process the data"""

        self.labels = Path(self.root).joinpath('points.json')
        self.labelsDict = self.loadLabelsJson()
        for i in range(1,len(self)):
            image,label=self.get(i)
            #convert label to a graph
            labelGraph=ShapeGraph(self.getPoints2DList(label))
            #create a data object
            data = Data(x=labelGraph.x, edge_index=labelGraph.edge_index,y=image)
            #save the data object
            torch.save(data, self.processed_paths[i-1])






    @property
    def raw_file_names(self):
        #get a list of all the files in the root directory with the extension .png
        return [f for f in Path(self.root).iterdir() if f.suffix=='.png']



    @property
    def processed_file_names(self):
        #processed files are saved as .pt files in the processed directory and
        #are named spoints.0001.pt, spoints.0002.pt etc
        return [f'spoints.{i:04d}.pt' for i in range(1,len(self.raw_file_names)+1)]



    def len(self):
        """len is number of entries in the labelsDict"""
        return len(self.processed_file_names)


    def get(self, idx):
        """get the data from the .pt files in the processed directory if file exists otherwise get the data from the raw directory"""
        if  Path(self.processed_dir).joinpath(self.processed_file_names[idx]).exists():
            data = torch.load(self.processed_paths[idx])
            return data.x, data.y
        else:
            #get the image
            image=read_image(self.raw_paths[idx])
            #get the label
            label=self.labelsDict[str(idx)]
            return image,label


class ellipsoid():
        """return an ellipse shape of Npoints"""
        def __init__(self):
            self.Npoints=10
            self.max_x=100
            self.max_y=100
            self.tan_len=10
            self.windowsize=224
            self.createShapeGraph()
        def createPoints(self):
            """given a set number of points create a set of x,y points on an ellipse shape"""
            points=[]
            for i in range(self.Npoints):
                x=math.sin(i*2*math.pi/self.Npoints)*self.max_x
                y=math.cos(i*2*math.pi/self.Npoints)*self.max_y
                points.append([x,y])
            #scale points to fit in a windowsize image
            points=[[x+self.windowsize/2,y+self.windowsize/2] for x,y in points]
            print(points)
            return points


        def createTangents(self):
            """create a list of left and right tangent handles in the format ([x,y],[x,y]) points based with a max value of tan_len"""
            tangents=[]
            for i in range(self.Npoints):
                x=math.sin(i*2*math.pi/self.Npoints)*self.tan_len
                y=math.cos(i*2*math.pi/self.Npoints)*self.tan_len
                tangents.append([[-x,-y],[x,y]])
            return tangents

        def createPoints2DList(self):
            """create on list of point2D objects"""
            points=self.createPoints()
            tangents=self.createTangents()
            return [point2D(points[i],tangents[i][0],tangents[i][1]) for i in range(self.Npoints)]

        def createShapeGraph(self):
            """create a shape graph object"""
            self.shape=ShapeGraph(self.createPoints2DList())








if __name__ == '__main__':
    #dataset = SimpleRotoDataset(root='D:/pyG/data/points/')
    #print(len(dataset))
    #print(dataset[1])
    e=ellipsoid()
    e.shape.printData()