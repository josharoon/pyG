

use_cuda = torch.cuda.is_available()

#get data loader

# get input ellipse splines

#create model wih input splines

#train model

for iters in range(1000):
    #train model
    image,spline=next(data_loader)
    #get loss
