#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Combined surface detection and flattened tiff creation notebook
# 8/1/2024
# ACdata must be mounted for access to zarr datasets


# In[3]:


import numpy
import json
import matplotlib.pyplot as plt
from tifffile import TiffFile,imread


# In[4]:


from surface_detection import get_zarr_group,detect_surface


# In[5]:


from skimage.io import imsave
def write_array_to_tiff(tiffpath,arr,exclude=None):
    for i in range(arr.shape[0]):
        if not (not exclude is None and i in exclude):
            imsave(tiffpath,arr[i,...],append=True,bigtiff=True)

# Function to get voxel resolution for a given mip level
def get_voxel_resolution(grp, mip_level):
    # Access the metadata
    metadata = grp.attrs.asdict()
    
    # Retrieve the voxel resolution for the specified mip level
    voxel_resolution = metadata['multiscales'][0]['datasets'][mip_level]['coordinateTransformations'][0]['scale']
    
    return voxel_resolution
            
def read_file(fpath):
    with TiffFile(fpath,is_ome=False) as tf:
        pg = tf.pages[0]
        data = numpy.zeros((pg.shape[0],pg.shape[1],len(tf.pages)),dtype=pg.dtype)
        for i,page in enumerate(tf.pages):
            data[:,:,i] = page.asarray()
    return data


# In[6]:


# materialization routine
# surface is a 2-D array of shape (data.shape[0],data.shape[1])
# of int values which are coordinate shifts to shift each column (axis 2)
# to flatten one surface of the volume
def materialize_tiff(tiffpath,data,surface,surfsup=False):
    A = data
    G = surface
    B = numpy.zeros(A.shape,dtype=A.dtype)
    xdim = B.shape[2]
    if not surfsup:        
        for z in range(A.shape[0]):
            for y in range(A.shape[1]):
                g = G[z,y]
                if g-10<0:
                    B[z,y,:] = A[z,y,:]
                else:
                    B[z,y,:xdim-(g-10)] = A[z,y,g-10:]
    else:
        for z in range(A.shape[0]):
            for y in range(A.shape[1]):
                g = G[z,y]
                if g+10 < xdim:
                    B[z,y,xdim-(g+10):] = A[z,y,:g+10]
                else:
                    B[z,y,:] = A[z,y,:]
    write_array_to_tiff(tiffpath,B.transpose((2,1,0)))


# In[7]:


# materialization routine
# surface is a 2-D array of shape (data.shape[0],data.shape[1])
# of int values which are coordinate shifts to shift each column (axis 2)
# to flatten one surface of the volume
def flatten_surface(data,surface,surfsup=False):
    A = data
    G = surface
    B = numpy.zeros(A.shape,dtype=A.dtype)
    xdim = B.shape[2]
    if not surfsup:        
        for z in range(A.shape[0]):
            for y in range(A.shape[1]):
                g = G[z,y]
                if g-10<0:
                    B[z,y,:] = A[z,y,:]
                else:
                    B[z,y,:xdim-(g-10)] = A[z,y,g-10:]
    else:
        for z in range(A.shape[0]):
            for y in range(A.shape[1]):
                g = G[z,y]
                if g+10 < xdim:
                    B[z,y,xdim-(g+10):] = A[z,y,:g+10]
                else:
                    B[z,y,:] = A[z,y,:]
    return B
    # write_array_to_tiff(tiffpath,B.transpose((2,1,0)))


# In[8]:


# get mip data from zarr
zarr = "/ACdata/Users/kevin/exaspim_ome_zarr/output_exa4/test.zarr/"
tile = "tile_x_0002_y_0001_z_0000_ch_488"
grp = get_zarr_group(zarr,tile)

# :,:,: is for entire tile, but cutout can be made by indexing on spatial axes
miplvl = 3
mipdata = grp[miplvl][0,0,:,:,:].transpose((2,1,0))


# In[9]:


resolution = get_voxel_resolution(grp, miplvl)
resolution = numpy.array([resolution[4],resolution[3],resolution[2]])
resolution


# In[10]:


mipdata.shape


# In[11]:


mipmax = numpy.max(mipdata,axis=2)


# In[12]:


plt.imshow(mipmax)


# In[13]:


# detect surface to flatten against (top: True, bottom: False)
# defines coordinate shift for every column of voxels to flatten one surface
surf_is_up = False
B = numpy.round(detect_surface(mipdata,surfsup=surf_is_up)).astype(int)


# In[ ]:


plt.imshow(B)
numpy.mean(B)


# In[ ]:


flatten_bottom = flatten_surface(mipdata,B.transpose(),surfsup=surf_is_up)


# In[ ]:


flatten_bottom.shape


# In[ ]:


# detect surface to flatten against (top: True, bottom: False)
# defines coordinate shift for every column of voxels to flatten one surface
surf_is_up = True
T = numpy.round(detect_surface(flatten_bottom,surfsup=surf_is_up)).astype(int)


# In[ ]:


plt.imshow(T)
numpy.mean(T)


# In[ ]:


materialize_tiff(tiffpath="flat_mip3.tif",data=flatten_bottom,surface=T.transpose(),surfsup=surf_is_up)


# In[ ]:




