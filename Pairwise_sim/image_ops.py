import numpy as np
from PIL import Image
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def combine_image_and_heatmap(img,heatmap):
    """
    Takes in a numpy array for an image and the similarity heatmap.
    Blends the two images together and returns a np array of the blended image.
    """
    cmap = plt.get_cmap('jet') # colormap for the heatmap
    heatmap = heatmap - np.min(heatmap)
    heatmap /= np.max(heatmap)
        # heatmap /= np.max(heatmap)-np.min(heatmap)
    heatmap = cmap(heatmap)
    # heatmap = cmap(np.max(heatmap)-heatmap)
    if np.max(heatmap) < 255.:
        heatmap *= 255
    # if np.max(heatmap) <= 1:
    #     heatmap *= 255
     # Blend image and heatmap
    if np.max(img) < 255:      # Added
        img *= 255
    # if np.max(img) <= 1:      # Added
    #     img *= 255
    img=np.tile(np.transpose(img,[1,2,0]), [1,1,3]) # Change (1,128, 128) into (128,128,3) # For lcnn9 
    # img=np.transpose(img,[1,2,0]) # Change (3,224,224) into (224,224,3) # For vgg16
    bg = Image.fromarray(img.astype('uint8')).convert('RGBA')
    fg = Image.fromarray(heatmap.astype('uint8')).convert('RGBA')
    fg=fg.resize(bg.size, Image.ANTIALIAS)    # Added
    outIm = np.array(Image.blend(bg,fg,alpha=0.5).convert('RGB'))  
    outIm=outIm[:,:,[-1 ,1 ,0]] # To convert RGB into BGR
    return outIm

def combine_horz(images):
    """
    Combines two images into a single side-by-side PIL image object.
    """
    images = [Image.fromarray(img.astype('uint8')[:,:,[-1,1,0]]) for img in images]  # To Convert RGB to BGR
#    images = [Image.fromarray(img.astype('uint8')) for img in images]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    return new_im


