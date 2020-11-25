#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import geopandas as gpd
import rasterio
#import tifffile as tiff
import matplotlib.pyplot as plt
from shapely import wkt
from shapely import affinity
import shapely
from descartes import PolygonPatch
import os
from rasterio import plot as rplot
import json
import numpy as np
import glob


def scale_image(img):
    '''
    scale the arrays within a rasterio datasetreader object to the range [0:1]
    '''
    img_scaled = np.empty(img.read().shape)
    for i,band in enumerate(img.read()):
        img_max = np.max(band)
        img_min = np.min(band)

        scaled_band = (band - img_min) * 1/(img_max-img_min)

        img_scaled[i] = scaled_band
        
    return img_scaled

def scale_geometry(x, img, img_id, grid_df):
    if len(img.shape)>2:
        _,h,w = img.shape
    else:
        h,w = img.shape
    xmax, ymax = grid_df.loc[img_id]
    
    W = w**2 / (w+1)
    H = h**2 / (h+1)

    if type(x) != shapely.geometry.multipolygon.MultiPolygon:
        x,y = np.array(x.exterior.coords.xy[0]), np.array(x.exterior.coords.xy[1])
        X = (x / xmax) * W
        Y = (y / ymax) * H
        scaled_coords = np.concatenate([X[:,None],Y[:,None]],axis=1)
        return shapely.geometry.Polygon(scaled_coords)
    else:
        polys = []
        for line in list(x.boundary.geoms):
                   
            x,y = np.array(line)[:,0], np.array(line)[:,1]
            X = (x / xmax) * W
            Y = (y / ymax) * H
            scaled_coords = np.concatenate([X[:,None],Y[:,None]],axis=1)
            polys.append(shapely.geometry.Polygon(scaled_coords))
        return shapely.geometry.MultiPolygon(polys)  
    
def get_labeled_polygons(image_id, grid_df):
    '''
    return dictionary of labeled polygons from geojson files in train_geojson_v4 directory
    '''
    fp = os.path.join('train_geojson_v3',image_id)
    
    if os.path.exists(fp):
        image = rasterio.open(os.path.join('three_band',image_id+'.tif'))
        image_scaled = scale_image(image)
        poly_dict = {}
        mapping_dict = {}

    #     for f in glob.glob(os.path.join(fp,'[!Grid]*.geojson')):
        for f in os.listdir(fp):
            if f[:4] == 'Grid':
                pass
            else:
                label = f[:-8]
                gdf = gpd.read_file(os.path.join(fp,f))
                gdf['geometry'] = gdf['geometry'].apply(scale_geometry,args=(image_scaled,image_id,grid_df))

                poly_dict[label] = gdf['geometry'].values
        image.close()
                
        return poly_dict
    else:
        print(f'{image_id}: is not a training image')
        
        
def plot_polys(spatial_dict, axes, geojson=True):
    '''
    plot image with layer of spatial objects from dictionary
    args:
        spatial_dict:  Keys are the class labels and values are polygons
        axes: matplotlib axes object on which to plot
    '''
    
    if not bool(spatial_dict):
        print("no polygons to plot")
    else:
        
        for class_label in spatial_dict.keys():
            if geojson:
                color_mapped = mapping_dict[class_label]
            else:
                color_mapped = class_label
            for geom in spatial_dict[class_label]:
                axes.add_patch(PolygonPatch(geom,alpha=0.65,ec=colors[color_mapped],fc='ivory'))
            
            
class_labels = {1:'Buildings',
                2:'Misc. Manmade structures',
                3:'Road', 
                4:'Track', 
                5:'Trees', 
                6:'Crops',
                7:'Waterway',
                8:'Standing water',
                9:'Vehicle Large',
                10:'Vehicle Small'}

colors = {1:'slategray',
          2:'lightsteelblue',
          3:'darkorange', 
          4:'red', 
          5:'green', 
          6:'wheat',
          7:'navy',
          8:'aqua',
          9:'deeppink',
          10:'fuchsia' }

mapping_dict = {'001_MM_L2_LARGE_BUILDING':1,
 '001_MM_L3_EXTRACTION_MINE':2,
 '001_MM_L3_NON_RESIDENTIAL_BUILDING':1,
 '001_MM_L3_RESIDENTIAL_BUILDING':1,
 '001_MM_L4_BRIDGE':2,
 '001_MM_L5_MISC_SMALL_STRUCTURE':2,
 '002_TR_L3_GOOD_ROADS':3,
 '002_TR_L4_POOR_DIRT_CART_TRACK':4,
 '002_TR_L6_FOOTPATH_TRAIL':4,
 '003_VH_L4_AQUATIC_SMALL':10,
 '003_VH_L4_LARGE_VEHICLE':9,
 '003_VH_L5_SMALL_VEHICLE':10,
 '003_VH_L6_MOTORBIKE':10,
 '004_UPI_L5_PYLONS':2,
 '004_UPI_L6_SATELLITE_DISHES_DISH_AERIAL':2,
 '005_VO_L6_MASTS_RADIO_TOWER':2,
 '005_VO_L7_FLAGPOLE':2,
 '006_VEG_L2_SCRUBLAND':5,
 '006_VEG_L2_WOODLAND':5,
 '006_VEG_L3_HEDGEROWS':5,
 '006_VEG_L5_GROUP_TREES':5,
 '006_VEG_L5_STANDALONE_TREES':5,
 '007_AGR_L2_CONTOUR_PLOUGHING_CROPLAND':6,
 '007_AGR_L2_DEMARCATED_NON_CROP_FIELD':6,
 '007_AGR_L2_ORCHARD':6,
 '007_AGR_L6_ROW_CROP':6,
 '007_AGR_L7_FARM_ANIMALS_IN_FIELD':6,
 '008_WTR_L2_STANDING_WATER':8,
 '008_WTR_L3_DRY_RIVERBED':7,
 '008_WTR_L3_WATERWAY':7,}