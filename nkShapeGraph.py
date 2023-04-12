"""This is an example data representation of a simple 4 point rotoshape which we then convert to a graph representation."""
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
import matplotlib.pyplot as plt

# Define the points of the rotoshape (tangent coordinates are relative to vertex coordinates)
# Point1    vertex coordinate{ 1150, 404, 0 } lftTang { -199.861, -74.1754, 0 }rhtTgt { 194, 72, 0 }
# Point2    vertex coordinate{ 1536, 750, 0 } lftTang { 0, -191.091, 0 }       rhtTgt { 0, 191.091, 0 }
# Point3    vertex coordinate{ 1150, 1096, 0} lftTang { 213.182, 0, 0 }        rhtTgt { -213.182, 0, 0 }
# Point4    vertex coordinate{ 764, 750, 0 }  lftTang { 0, 191.091, 0 }        rhtTgt  { 0, -191.091, 0 }

class point2D:
    def __init__(self, vertex, lftTang=None, rhtTang=None):

        self.vertex = vertex[:2]
        if lftTang is None:
              self.lftTang = None
        else:
            self.lftTang = lftTang[:2]
        if rhtTang is None:
            self.rhtTang = None
        else:
            self.rhtTang = rhtTang[:2]




class ShapeGraph:
    """Points are represented as Nodes with edges between them."""
    def __init__(self, points):
        """take an ordered list of points and create a graph representation of the shape"""
        self.points = points
        self.nPoints = len(points)
        self.edges=[]
        self.x=[]
        self.createEdgeIndex()
        self.createX()
        self.createData()



    def createEdgeIndex(self):
        """create the edge index from the graph representation of the shape"""
        for i in range(self.nPoints):
            self.edges.append([i, (i + 1) % self.nPoints])
            #add an edge from the last point to the first point
        #self.edges.append([self.nPoints - 1, 0])
        self.edge_index = torch.tensor(self.edges, dtype=torch.long).t().contiguous()

    def createX(self):
        """for each node convert point into a vector of 6 values and add to the x list"""
        for point in self.points:
            self.x.append(point.vertex + point.lftTang + point.rhtTang)
        self.x = torch.tensor(self.x, dtype=torch.float)


    def createData(self):
        """create a data object from X and edge_index"""
        self.data = Data(x=self.x, edge_index=self.edge_index)



    def createGraph(self):
        """create a graph representation of the shape"""
        self.G = nx.Graph()
        for i, point in enumerate(self.points):
            self.G.add_node(i, vertex=point.vertex, lftTang=point.lftTang, rhtTang=point.rhtTang)
        for i in range(self.nPoints):
            self.G.add_edge(i, (i + 1) % self.nPoints)
        #add an edge from the last point to the first point
        self.G.add_edge(self.nPoints - 1, 0)

    def createNxGraph(self):
        """convert the data object to a networkx graph"""
        self.nxG = to_networkx(self.data)





    def drawGraph(self):
        """draw the graph representation of the shape with point values in the node labels"""
        self.createNxGraph()
        nx.draw(self.nxG, with_labels=True, font_weight='bold')
        plt.show()

    def printGraph(self):
        """print the graph representation of the shape with point values in the node labels"""
        print(self.nxG.nodes.data())

    def printEdges(self):
        """print the graph representation of the shape with point values in the node labels"""
        print(self.nxG.edges.data())

    def printData(self):
        """print the graph representation of the shape with point values in the node labels"""
        print(f"keys: {self.data.keys}, edge_index: {self.data.edge_index}, x: {self.data.x} \n  nodeFeatures: {self.data.num_node_features}, numNodes: {self.data.num_nodes}, numEdges: {self.data.num_edges}")



# inherit from shape graph with tangents considered as extra vertices rather than features
class ShapeGraphTangents(ShapeGraph):
    def __init__(self, points):
        """take an ordered list of points and create a graph representation of the shape"""
        self.points = points
        self.nPoints = len(points)
        self.edges=[]
        self.x=[]
        self.createEdgeIndex()
        self.createX()
        self.createData()

    def createEdgeIndex(self):
        """create the edge index from the graph representation of the shape"""
        for i in range(self.nPoints):
            self.edges.append([i, (i + 1) % self.nPoints])
            #add an edge from the last point to the first point
        #self.edges.append([self.nPoints - 1, 0])
        self.edge_index = torch.tensor(self.edges, dtype=torch.long).t().contiguous()

    def createX(self):
        """for each node convert point into a vector of 6 values and add to the x list"""
        for point in self.points:
            self.x.append(point.vertex)
        self.x = torch.tensor(self.x, dtype=torch.float)



if __name__ == '__main__':
    point = point2D([1150, 404, 0], [-199.861, -74.1754, 0], [194, 72, 0])
    point2= point2D([1536, 750, 0], [0, -191.091, 0], [0, 191.091, 0])
    point3= point2D([1150, 1096, 0], [213.182, 0, 0], [-213.182, 0, 0])
    point4= point2D([764, 750, 0], [0, 191.091, 0], [0, -191.091, 0])
    points = [point, point2, point3, point4]
    shape = ShapeGraph(points)
    shape2 = ShapeGraphTangents(points)
    shape.printData()
    shape.drawGraph()
    shape.printGraph()
    shape.printEdges()


