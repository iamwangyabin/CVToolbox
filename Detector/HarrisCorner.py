from scipy.ndimage import filters
import numpy as np
from PIL import Image
import matplotlib
import pylab

def compute_harris_response(im,sigma=3):
    imx=np.zeros(im.shape)
    filters.gaussian_filter(im,(sigma,sigma),(0,1),imx)
    imy=np.zeros(im.shape)
    filters.gaussian_filter(im,(sigma,sigma),(1,0),imy)

    Wxx=filters.gaussian_filter(imx*imx,sigma)
    Wxy=filters.gaussian_filter(imx*imy,sigma)
    Wyy=filters.gaussian_filter(imy*imy,sigma)

    Wdet=Wxx*Wyy-Wxy**2
    Wtr=Wxx+Wyy

    return Wdet/Wtr

def get_harris_points(harrisim,min_dist=10,threshold=0.1):
    corner_threshold=harrisim.max()*threshold
    harrisim_t=(harrisim>corner_threshold)*1

    coords=np.array(harrisim_t.nonzero()).T

    candidate_values=[harrisim[c[0],c[1]] for c in coords]
    index=np.argsort(candidate_values)

    allowed_locations=np.zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist,min_dist:-min_dist]=1

    filtered_coords=[]
    for i in index:
        if allowed_locations[coords[i,0],coords[i,1]]==1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist),(coords[i,1]-min_dist):(coords[i,1]+min_dist)]
    return filtered_coords

def plot_harris_points(image,filtered_coords):
    pylab.figure()
    pylab.gray()
    pylab.imshow(image)
    pylab.plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords],"*", color="r")
    pylab.axis("off")
    pylab.gca().set_axis_off()
    pylab.subplots_adjust(top=1, bottom=0, right=1, left=0,
                    hspace=0, wspace=0)
    pylab.margins(0, 0)
    pylab.gca().xaxis.set_major_locator(pylab.NullLocator())
    pylab.gca().yaxis.set_major_locator(pylab.NullLocator())
    pylab.savefig("_1.jpg", bbox_inches='tight',
            pad_inches=0,dpi=300)
    pylab.show()

def harris_points(image,filtered_coords):
    point=([p[1] for p in filtered_coords],[p[0] for p in filtered_coords])
    return point

if __name__ == '__main__':
    im=np.array(Image.open('1.jpg').convert('L'))
    harrisim=compute_harris_response(im)
    filtered_coords=get_harris_points(harrisim,6)
    plot_harris_points(im,filtered_coords)