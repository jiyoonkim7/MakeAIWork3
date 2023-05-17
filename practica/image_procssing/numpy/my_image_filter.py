import io
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from skimage.measure import block_reduce # geimporteerd om methode 'downSample' te kunnen uitvoeren

class MyImageFilter(object):
    
    def __init__(self, kernel=None): #oorspr werd alleen 'kernel' genoemd. Dit geeft problemen als je downSample wilt uitvoeren. Daarom '=None en 'if kernel is...' toegevoegd
        if kernel is not None:
            
            self.imgKernel = kernel
    
    def convolve(self, imgTensor):
        imgTensorRGB = imgTensor.copy() 
        outputImgRGB = np.empty_like(imgTensorRGB)

        for dim in range(imgTensorRGB.shape[-1]):  # loop over rgb channels
            outputImgRGB[:, :, dim] = sp.signal.convolve2d (
                imgTensorRGB[:, :, dim], self.imgKernel, mode="same", boundary="symm"
            )
    
        return outputImgRGB        
    
    
    def downSample(self, imgTensor):
        return block_reduce(imgTensor, block_size=(2,2,1), func=np.max)
    
   #lengte, breedte, kleur
   #np.max kan ook np.mean of sum of alles door de helft delen  np.average
   #img2 = test.downSample(imOrg,np.average)
   #PlotImges(img2, img3+img2)