import cairo
import torch
from PIL import Image
import simpleRotoDataset
from torchmetrics.classification import JaccardIndex,dice
from torchvision.transforms import PILToTensor, ToPILImage




def splineArray2Image(points,color=(1, 1, 1), line_width=1, filename=r'D:\pyG\data\temp\shape.png'):
    #add tangent handles cordinates to xy points
    points[:,2:4]=points[:,0:2]+points[:,2:4]
    points[:,4:6]=points[:,0:2]+points[:,4:6]
    #swap left and right tangent handles
    points[:,2:4],points[:,4:6]=points[:,4:6],points[:,2:4]
    # #convert all y co-ordinates to 480-y
    points[:,1]=224-points[:,1]
    points[:,3]=224-points[:,3]
    points[:,5]=224-points[:,5]


    # Set the color and line width
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, 224, 224)
    context = cairo.Context(surface)
    context.set_source_rgb(*color)  # set color to red
    context.set_line_width(line_width)  # set line width to 2

    # Move to the first point
    x, y = points[0][0:2]
    context.move_to(x, y)
    #print(f"starting point: {x,y,context.get_current_point()}")
    # Draw the Bezier curves using the control points
    for i in range(1, len(points)):
        x, y= points[i][0:2]
        lx,ly=points[i-1][2:4]
        rx,ry=points[i-1][4:6]
        context.curve_to(lx, ly, rx, ry, x, y)
        #print(f"point: {x,y,context.get_current_point()}")
        #context.stroke()

    x, y = points[0][0:2]
    lx, ly = points[-1][2:4]
    rx, ry = points[-1][4:6]
    context.curve_to(lx, ly, rx, ry, x, y)
    #print(f"final point: {x,y}")


    #Close the path and fill the shape
    context.close_path()
    context.fill()
    # then draw the tangents handles in red
    # context.set_source_rgb(1, 0, 0)  # set color to red
    # for i in range(1, len(points)):
    #     x, y, lx, ly, rx, ry = points[i]
    #     context.move_to(x, y)
    #     context.line_to(lx, ly)
    #     context.move_to(x, y)
    #     context.line_to(rx, ry)
    #     context.stroke()

    #convert to PIL Image
    img =  Image.frombuffer("RGB", (surface.get_width(),
surface.get_height()), surface.get_data(), "raw", "RGBA", 0,1)
    #display the image
    #img.show()
    img=PILToTensor()(img)
    img=torch.where(img>0.5,1,0)
    if img.shape[0]==4:
        img=img[0:3,:,:]
    return img


if __name__ == '__main__':
    #create a 5 sided polygon shape and set tangent handels to 0
    # points = torch.tensor([
    #     [323, 387, 423, 387, 522, 335],
    #     [522, 235, 522, 235, 323, 79],
    #     [323, 79,323, 79, 130, 235],
    #     [130, 235,130, 235, 223, 387]
    # ])
    # #convert all y co-ordinates to 480-y
    # points[:,1]=480-points[:,1]
    # points[:,3]=480-points[:,3]
    # points[:,5]=480-points[:,5]
    #
    #
    # splineArray2Image(points)
    #grab some points from simple roto dataset
    dataset=simpleRotoDataset.SimpleRotoDataset(root='D:/pyG/data/points/',labelsJson="points310323_205433.json")
    gt,points=dataset[1634][:2]



    img2=splineArray2Image(points)



    #jack=JaccardIndex(task='binary')
    #print(jack(gt,img2))
    dice=dice.Dice()
    print(dice(gt,img2))

    #now make a comparison where gt&img2=1 set value to 0.5 where either is 1 set value to 1, where both are 0 set value to 0
    comp=torch.where(((gt==1)^(img2==1)),1,0)

    #convert comp to uint8 values and then to PIL image
    comp2=comp*255
    comp2=comp2.type(torch.uint8)
    comp2=ToPILImage()(comp2)


    #show both img and gt
    #now concatonate the gt and img2 (to double the width)
    gt2=torch.cat((img2,gt),dim=2)
    #now
    #convert gt to pill image and sho
    #convert gt to uint8 values and then to PIL image
    gt2=gt2*255
    gt2=gt2.type(torch.uint8)
    gt2=ToPILImage()(gt2)

    gt2.show()
    comp2.show()


    #exit(0)

