import os
import argparse

import numpy as np




def convexhull2D(points):
    """
    Returns the convex hull of a set of 2D points
    Parameters
    ----------
    points
        array of 2D points
    """
    points = np.array(points)
    if points.shape[1] != 2:
        print(f"Points aren't 2D, shape: {points.shape}")
    
    # gift wrapping algorithm O(n*h) (with h amount of points on hull)

    # TODO: implement O(n log n) recursive algorithm?

    
