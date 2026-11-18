import numpy as np
from pyproj import Transformer, CRS

def utm_crs_for(lon, lat):
    zone = int((lon+180)//6) + 1
    epsg = (32600 if lat>=0 else 32700) + zone
    return CRS.from_epsg(epsg)

def make_projector(lons, lats):
    clon, clat = float(np.mean(lons)), float(np.mean(lats))
    crs = utm_crs_for(clon, clat)
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    
    def project(lon, lat):
        return tr.transform(lon, lat)
    
    return project, crs