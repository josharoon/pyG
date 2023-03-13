"""simpleRotoDataset.py - A simple dataset for testing and training rotoscoping models."""
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



if __name__ == '__main__':
    dataset = SimpleRotoDataset(root='D:/pyG/data/points/')
    print(len(dataset))
    print(dataset[1])