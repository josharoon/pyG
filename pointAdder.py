"""loads shapes from simple roto dataset and adds extra points to the shapes"""
import matplotlib
import torch
from matplotlib import pyplot as plt
from torch_geometric.data import Data
from simpleRotoDataset import SimpleRotoDataset
from p2mUtils.utils import *
import cv2 as cv


class pointAdder():
    def __init__(self, dataset, num_points=10):
        self.dataset = dataset
        self.num_points = num_points

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        self.data = self.dataset[idx]
        self.contourPoints=self.predictContours(idx)
        self.splinePoints=self.tensor2Point2D(self.data[1])
        self.keyPointsIdx=self.getKeyPoints()
        self.pyGdata=self.getPyGData()
        return (self.data,self.contourPoints,self.splinePoints,self.keyPointsIdx,self.pyGdata)

    def predictContours(self,dataasetIdx):
        """predict the contours using openCV"""
        #get data item [0] (a tensor) and convert to cv2 image for contour detection
        self.image = self.data[dataasetIdx].numpy()
        #premute the dimensions to get the correct shape for cv2
        self.image = np.transpose(self.image, (1, 2, 0))
        self.image = cv.cvtColor(self.image, cv.COLOR_RGB2GRAY)
        #get the contours
        contours, hierarchy = cv.findContours(self.image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        #convert to an ordered list of point2D objects (set tangents to 0)
        #contours needs to be converted from 0,0 in top left to 0,0 in bottom left

        points = [point2D([x[0][0],224-x[0][1]]) for x in contours[0]]
        return points

    def tensor2Point2D(self,tensor):
        """convert a tensor to a point2D object"""
        # n*m tensor where n is the number of points and m represents x,y and tangents
        #convert to list of point2D objects
        points=[point2D([x[0],x[1]],[x[2],x[3]],[x[4],x[5]]) for x in tensor]
        return points

    def point2D2Tensor(self,points):
        """convert a list of point2D objects to a tensor discarding tangents"""
        #convert to tensor
        tensor=torch.tensor([[x.vertex[0],x.vertex[1]] for x in points])
        return tensor

    def getKeyPoints(self):
        """find the index of the key points in contourPoints that are closest to the spline points"""
        #find the index of the key points in contourPoints that are closest to the spline points
        keyPointsIdx=[]
        for point in self.splinePoints:
            #find the index of the closest point in contourPoints
            idx = min(range(len(self.contourPoints)), key=lambda i: self.distanceP2D(self.contourPoints[i],point))
            keyPointsIdx.append(idx)
        return keyPointsIdx

    def distanceP2D(self,p1,p2):
        """calculate the distance between two point 2D objects"""
        return np.sqrt((p1.vertex[0]-p2.vertex[0])**2+(p1.vertex[1]-p2.vertex[1])**2)

    def getPyGData(self):
        """convert the data to a pyG data object"""
        "we add edges between all the contour points, we set edges weights to [0,0] between all contour points and " \
        "use tangent values from the spline points to create edges between the key points"
        #create a tensor of edge index values from contour points in bidirectional order e.g [[0,1],[1,0],[1,2],[2,1]...]
        edge_index = torch.tensor([[i, i + 1] for i in range(len(self.contourPoints))], dtype=torch.long).contiguous()
        edge_index2=edge_index.flip(1)
        #interleave the two tensors to get the bidirectional edges get get edge index[0] then edge index2[1]
        edge_index3= torch.zeros((len(edge_index)*2,2),dtype=torch.long)
        edge_index3[::2]=edge_index
        edge_index3[1::2]=edge_index2
        #set edge attributes for all edges to [0,0]
        edge_attr = torch.zeros((len(edge_index)*2, 2), dtype=torch.float)
        #set the edge attributes for the edges between the key points to the tangent values
        #use left tangent for edge attr in clockwise direction and right tangent for edge attr in anticlockwise direction
        for i in range(len(self.keyPointsIdx)-1):
            #get the index of the key points
            idx1=self.keyPointsIdx[i]*2
            idx2=idx1+1
            #get the tangent values
            leftTangent=self.splinePoints[i].lftTang
            rightTangent=self.splinePoints[i].rhtTang
            #set the edge attributes
            edge_attr[idx1]=torch.tensor(leftTangent,dtype=torch.float)
            edge_attr[idx2]=torch.tensor(rightTangent,dtype=torch.float)
        #at the key points set the edge attributes to the tangent values
        pygData = Data(x=torch.tensor(self.point2D2Tensor(self.contourPoints), dtype=torch.float), edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)
        return pygData








if __name__ == "__main__":
    dataset = SimpleRotoDataset(root=r"D:\pyG\data\points\120423_183451_rev", labelsJson="points120423_183451_rev.json")
    pointAdder = pointAdder(dataset)
    data = pointAdder[0]
    print(data)
    #plot the points from the pyG data object in blue and spline points in red, the key points are in green, and the edge attributes are in black
    pyGdata=data[4]
    splinePoints=data[2]
    keyPointsIdx=data[3]
    #get list of x.y values for the spline points
    x=[x.vertex[0] for x in splinePoints]
    y=[x.vertex[1] for x in splinePoints]
    #get list of x,y values from the PyG data object
    x2=pyGdata.x[:,0].tolist()
    y2=pyGdata.x[:,1].tolist()
    #get the edge attributes and keypoints indices
    edgeAttr=pyGdata.edge_attr
    #doulbe the values in the key points index list to get the index of the edge attributes
    #get the x,y values for the edge attributes at the key points
    x3=[edgeAttr[i*2][0]+x2[i] for i in keyPointsIdx]
    y3=[edgeAttr[i*2][1]+y2[i] for i in keyPointsIdx]
    x4=[edgeAttr[i*2+1][0]+x2[i] for i in keyPointsIdx]
    y4=[edgeAttr[i*2+1][1]+y2[i] for i in keyPointsIdx]


    #plot the points in a 224x224 image
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, c='r')
    plt.scatter(x2, y2, c='b')
    plt.scatter([x2[i] for i in keyPointsIdx], [y2[i] for i in keyPointsIdx], c='g')
    plt.scatter(x3, y3, c='k')
    plt.scatter(x4, y4, c='k')
    plt.xlim(0, 224)
    plt.ylim(0, 224)
    #matplotlib.use('WebAgg')
    plt.interactive(False)

    plt.show()
    #show the image
    plt.imshow(data[0][0].numpy().transpose(1, 2, 0))
    plt.show()




