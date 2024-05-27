# -*- coding:utf-8 -*-

import numpy as np
import networkx as nx

def screen_by_hist(x_data, y_data, bins):
    extenti = (x_data.min()-0.1, x_data.max()+0.1)
    extentj = (y_data.min()-0.1, y_data.max()+0.1)
    hist, edges_x, edges_y = np.histogram2d(
        x_data, y_data, bins=bins,range=(extenti, extentj)
    )
    edges_x = (edges_x[:-1] + edges_x[1:]) / 2
    edges_y = (edges_y[:-1] + edges_y[1:]) / 2
    
    cell_idXy = np.stack((np.arange(hist.shape[0]),hist.argmax(axis=1)),axis=-1)
    cell_idYx = np.stack((hist.argmax(axis=0),np.arange(hist.shape[1])),axis=-1)
    cell_idxy = np.vstack((cell_idXy,cell_idYx))
    cell_idxy = np.unique(cell_idxy,axis=0)
    cell_idxy = cell_idxy[~(cell_idxy == 0).any(axis=1)]
    
    x_hist = edges_x[cell_idxy[:,0]]
    y_hist = edges_y[cell_idxy[:,1]]
    cell_counts = hist[tuple(cell_idxy.T)]

    rho=cell_counts
    if np.max(rho)-np.min(rho)!=0:
        rho = (rho - np.min(rho))/(np.max(rho)-np.min(rho))+1
    return x_hist, y_hist, rho

def screen_by_graph(x_screen1, y_screen1, rho):
    G = nx.DiGraph()
    for i in range(len(x_screen1)):
        x_curr, y_curr = x_screen1[i], y_screen1[i]
        candidates_idx = (x_screen1 >= x_curr) & (y_screen1 >= y_curr)
        candidates_idx[i] = False
        
        x_candidates = x_screen1[candidates_idx]
        y_candidates = y_screen1[candidates_idx]
        rho_candidates = rho[candidates_idx]

        candidates = [((x_curr, y_curr), (x, y),{"edge":rho[i]*r/((x_curr-x)**2+(y_curr-y)**2)**0.5}) 
                      for x, y ,r in zip(x_candidates, y_candidates, rho_candidates)]
        G.add_edges_from(candidates)

    longest_path = nx.dag_longest_path(G,weight="edge")
    x_screen2, y_screen2 = zip(*longest_path)
    x_screen2, y_screen2 = np.array(x_screen2), np.array(y_screen2)

    return x_screen2, y_screen2

def polish_ends(x_screen2, y_screen2, tol_bins):
    center_idx = int(len(x_screen2) / 4)

    x, y = x_screen2[:center_idx], y_screen2[:center_idx]
    stepx = x[1:] - x[:-1]
    good_x = (stepx / stepx[stepx > 0].min()) < tol_bins
    stepy = y[1:] - y[:-1]
    good_y = (stepy / stepy[stepy > 0].min()) < tol_bins
    good_xy = good_x & good_y
    breaks_idx = np.where(good_xy == False)[0]
    
    break_idx = 0
    if len(breaks_idx) > 0:
        idx = np.where(breaks_idx < len(x) * 0.25)[0]
        if len(idx) > 0:
            break_idx = breaks_idx[idx][-1] + 1
    x_left, y_left = x[break_idx:], y[break_idx:]

    x, y = x_screen2[center_idx:], y_screen2[center_idx:]
    stepx = x[1:] - x[:-1]
    good_x = (stepx / stepx[stepx > 0].min()) < tol_bins
    stepy = y[1:] - y[:-1]
    good_y = (stepy / stepy[stepy > 0].min()) < tol_bins
    good_xy = good_x & good_y
    breaks_idx = np.where(good_xy == False)[0]
    
    break_idx = len(x)
    if len(breaks_idx) > 0:
        idx = np.where(breaks_idx > len(x) * 0.75)[0] 
        if len(idx) > 0:
            break_idx = breaks_idx[idx][0] + 1 
    x_right, y_right = x[:break_idx], y[:break_idx]
    
    x = np.concatenate([x_left, x_right])
    y = np.concatenate([y_left, y_right])
    return x, y