"""simpleRotoDataset.py - A simple dataset for testing and training rotoscoping models."""
import math
import re

import torch
from pathlib import Path
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.utils import to_networkx
from torchvision.io import read_image
from torchvision.transforms import ToPILImage
from nkShapeGraph import ShapeGraph,point2D,ShapeGraphTangents
from p2mUtils.utils import *
from p2mUtils.viz import plot_distance_field
from tqdm import tqdm
import os
import concurrent.futures
import matplotlib.pyplot as plt
import glob
#from ImageGraph import ImageGraph
import json


class SimpleRotoDataset(Dataset):
    """A simple dataset for testing and training rotoscoping models."""
    def  __init__(self, root,labelsJson, transform=None, pre_transform=None,pre_filter=None, preprocess=True):
        self.labelsFile=labelsJson
        super().__init__(root, transform, pre_transform,pre_filter)
        self.root=root
        self.preprocess = preprocess


    def normalize_image(self, image):
        return image / 255.0

    def normalize_distance_field(self, distance_field):
        d_max = torch.max(distance_field)
        return distance_field / d_max



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
        points2D = [self.Shape2Point2D(shape) for shape in pointList]
        # for point in points2D:
        #     point.normalize
        #add center point to tangent handels in each point
        for point in points2D:
            point.lftTang[0]=point.lftTang[0]=point.lftTang[0]+point.vertex[0]
            point.lftTang[1]=point.lftTang[1]=point.lftTang[1]+point.vertex[1]
            point.rhtTang[0]=point.rhtTang[0]=point.rhtTang[0]+point.vertex[0]
            point.rhtTang[1]=point.rhtTang[1]=point.rhtTang[1]+point.vertex[1]

        return points2D

    def process_image(self, index):
        image, label = self.get(index)
        image = self.normalize_image(image)
        labelGraph = ShapeGraph(self.getPoints2DList(label))
        #we scale everything to 224x224 for the model
        original_height, original_width = image.shape[1:]
        if original_height != 224 or original_width != 224:
            # Calculate scaling factors
            height_scale_factor = 224 / original_height
            width_scale_factor = 224 / original_width
            # Resize the image
            image = th.nn.functional.interpolate(
                image.unsqueeze(0), size=(224, 224), mode="bilinear"
            ).squeeze(0)
            # Scale the control points
            for point in labelGraph.points:
               point.scalePoints(width_scale_factor,height_scale_factor)
            labelGraph.x = []
            labelGraph.createX()
            labelGraph.createData()



        data = Data(x=labelGraph.x, edge_index=labelGraph.edge_index, y=image)
        control_points = convert_to_cubic_control_points(data.x[None, :]).to(th.float64)
        grid_size = data.y.shape[1]
        source_points = create_grid_points(grid_size, 0, 224, 0, 224)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        control_points = control_points.to(device)
        source_points = source_points.to(device)

        distance_field = distance_to_curves(source_points, control_points, grid_size).view(grid_size, grid_size)
        distance_field = th.flip(distance_field, (1,0))
        distance_field = self.normalize_distance_field(distance_field)

        torch.save(data, self.processed_paths[index])
        torch.save(distance_field, Path(self.processed_dir).joinpath(f'distance_field.{index+1:04d}.pt'))


    def process(self):
        self.labels = Path(self.root).joinpath(self.labelsFile)
        self.labelsDict = self.loadLabelsJson()

        num_workers = 2  # Adjust this according to your system's resources

        # Find all processed data files using glob
        processed_files = set(glob.glob(os.path.join(self.processed_dir, '*.pt')))
        pattern = re.compile(r'.*\.(\d+)\.pt')
        processed_indices = {int(pattern.match(os.path.basename(f)).group(1)) for f in processed_files}
        #we need to subtract 1 from each index because the indices in the json file start at 1
        processed_indices={x-1 for x in processed_indices}
        print(f"Found {len(processed_files)} processed files.")

        print(f"Found {len(processed_indices)} processed indices.")

        # Calculate unprocessed indices
        print("Calculating unprocessed indices...")
        unprocessed_indices = [i for i in range(len(self)) if i not in processed_indices]
        print(f"Found {len(unprocessed_indices)} unprocessed files.")
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            list(tqdm(executor.map(self.process_image, unprocessed_indices), total=len(unprocessed_indices),
                      desc="Processing",
                      unit="image"))

    @property
    def raw_file_names(self):
        #get a list of all the files in the root directory with the extension .png
        paths=[f for f in Path(self.root).iterdir() if f.suffix=='.png']
        #makes sure the files are sorted by name by number e.g. spoints.0001.png, spoints.0002.png take into account varying number of digits
        paths.sort(key=lambda f: int(''.join(filter(str.isdigit, f.name))))
        return paths



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
            df=torch.load(Path(self.processed_dir).joinpath(f'distance_field.{idx+1:04d}.pt'))
            return data.y, data.x , self.processed_paths[idx],df
        else:
            #get the image
            image=read_image(self.raw_paths[idx])
            #get the label
            label=self.labelsDict[str(idx+1)]
            return image,label

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self):
            result = self.get(self.n)
            self.n += 1
            return result
        else:
            raise StopIteration




class ellipsoid():
        """return an ellipse shape of Npoints"""
        def __init__(self, npoints=5):
            self.Npoints= npoints
            self.max_x=100
            self.max_y=100
            self.tan_len=10
            self.windowsize=224
            self.shape=None
            self.createShapeGraph()
            self.getChebPolys()
            self.getVertexNeighbours()
            self.lapIndex=torch.as_tensor(cal_lap_index(self.vertexNeighbours))

        def getVertexNeighbours(self):
            """get the neighbours of each vertex and return asa list of lists"""
            #iterate through edge index to get the neighbours of each vertex
            neighbours=[]
            for p in range(self.Npoints):
                if p==0:
                    before=self.shape.edges[-1][0]
                else:
                    before=self.shape.edges[p-1][0]
                if p==self.Npoints-1:
                    after=self.shape.edges[0][0]
                else:
                    after=self.shape.edges[p+1][0]
                neighbours.append([before,after])
            self.vertexNeighbours=neighbours
            print(self.vertexNeighbours)









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

        def getChebPolys(self):
            """return the chebyshev polynomials for the shape"""
            netxData = to_networkx(self.shape.data, to_undirected=True)
            adj = nx.adjacency_matrix(netxData)
            self.cheb = chebyshev_polynomials(adj, 1)


        def plotPoints(self):
                """plot the points of the  elllipse using matplotlib in a 224 x 224 grid with their index number"""
                points=self.shape.x
                plt.figure(figsize=(10,10))
                plt.xlim(0,224)
                plt.ylim(0,224)
                plt.scatter(points[:,0],points[:,1])
                for i in range(self.Npoints):
                    plt.text(points[i,0],points[i,1],str(i))
                plt.show()



#ellipsoid class using shapeGraphTangent class
class ellipsoid2(ellipsoid):
    """same as ellipsoid class but using the shapeGraphTangent class"""
    def createPoints2DList(self):
        """create on list of point2D objects"""
        points=self.createPoints()
        tangents=self.createTangents()
        #iterate through points and tangents returning a list of point2D objects with tangents as seperate point2D objects
        points2DList=[]
        for i in range(self.Npoints):
            points2DList.append(point2D(points[i]))
            points2DList.append(point2D([tangents[i][0][0]+points[i][0],tangents[i][0][1]+points[i][1]]))
            points2DList.append(point2D([tangents[i][1][0]+points[i][0],tangents[i][1][1]+points[i][1]]))
        self.Npoints=len(points2DList)
        return points2DList


    def createShapeGraph(self):
        """create a shape graph object"""
        self.shape=ShapeGraphTangents(self.createPoints2DList())







if __name__ == '__main__':
    from dfUtils.cubicCurvesUtil import *

    #dataset = SimpleRotoDataset(root=r'D:\pyG\data\points\120423_183451_rev', labelsJson="points120423_183451_rev.json")
    dataset = SimpleRotoDataset(root=r'D:\pyG\data\points\transform_test', labelsJson="pointstransform_test.json")
    # print(len(dataset))
    # print(dataset[0])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    dataIter = iter(dataloader)
    data = next(dataIter)
    #
    #
    #
    # plot cubic spline from ground truth
    control_points = convert_to_cubic_control_points(data[1])
    plotCubicSpline(control_points)

    plt.show()







0