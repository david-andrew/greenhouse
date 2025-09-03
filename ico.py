import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def icosahedron():
    """Return (V, F): 12 vertices and 20 triangular faces (indexing into V)."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    V = np.array([
        (-1,  phi, 0), ( 1,  phi, 0), (-1, -phi, 0), ( 1, -phi, 0),
        (0, -1,  phi), (0,  1,  phi), (0, -1, -phi), (0,  1, -phi),
        ( phi, 0, -1), ( phi, 0,  1), (-phi, 0, -1), (-phi, 0,  1)
    ], dtype=float)

    F = np.array([
        (0,11,5), (0,5,1), (0,1,7), (0,7,10), (0,10,11),
        (1,5,9), (5,11,4), (11,10,2), (10,7,6), (7,1,8),
        (3,9,4), (3,4,2), (3,2,6), (3,6,8), (3,8,9),
        (4,9,5), (2,4,11), (6,2,10), (8,6,7), (9,8,1)
    ], dtype=int)
    return V, F

def set_axes_equal(ax):
    x_limits, y_limits, z_limits = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    x_range, y_range, z_range = abs(x_limits[1]-x_limits[0]), abs(y_limits[1]-y_limits[0]), abs(z_limits[1]-z_limits[0])
    x_mid, y_mid, z_mid = np.mean(x_limits), np.mean(y_limits), np.mean(z_limits)
    r = 0.5*max([x_range, y_range, z_range])
    ax.set_xlim3d([x_mid-r, x_mid+r]); ax.set_ylim3d([y_mid-r, y_mid+r]); ax.set_zlim3d([z_mid-r, z_mid+r])

def subdivided_icosahedron_points_edges(N):
    """
    N = number of interior points inserted along each original edge
        (i.e., edge is split into N+1 segments).
    Returns:
      P_unit: (V,3) points on the unit sphere
      edges:  list of (i,j) index pairs into P_unit
    """
    V, F = icosahedron()
    f = N + 1
    point_index, points, edges_set = {}, [], set()

    def idx_for_point(p):
        key = tuple(np.round(p, 12))
        if key in point_index: return point_index[key]
        i = len(points); point_index[key] = i; points.append(p); return i

    for (i0, i1, i2) in F:
        v0, v1, v2 = V[i0], V[i1], V[i2]
        local = {}
        # Generate barycentric grid points with i+j+k=f
        for a in range(f+1):
            for b in range(f-a+1):
                c = f - a - b
                p = (a*v1 + b*v2 + c*v0) / float(f)
                local[(a,b)] = idx_for_point(p)

        # Triangular tiling edges
        for a in range(f):
            for b in range(f - a):
                A = local[(a, b)]
                B = local[(a+1, b)]
                C = local[(a, b+1)]
                for u, v in ((A,B),(A,C),(B,C)): edges_set.add(tuple(sorted((u,v))))
                if a + b < f - 1:
                    D = local[(a+1, b+1)]
                    for u, v in ((B,D),(C,D),(B,C)): edges_set.add(tuple(sorted((u,v))))

    P = np.asarray(points, float)
    norms = np.linalg.norm(P, axis=1, keepdims=True); norms[norms==0.0] = 1.0
    P_unit = P / norms
    return P_unit, sorted(edges_set)

def plot_icosphere(P, edges, show_points=True, point_size=100):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    # Ensure equal aspect ratio so the sphere isn't squished
    ax.set_box_aspect((1,1,1))
    if show_points: ax.scatter(P[:,0], P[:,1], P[:,2], s=50.0, color='r')
    for i, j in edges:
        ax.plot([P[i,0],P[j,0]], [P[i,1],P[j,1]], [P[i,2],P[j,2]], color='k', linewidth=1.2)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    set_axes_equal(ax)
    ax.set_title('Subdivided Icosahedron (Icosphere)')
    plt.show()

# ---- Example run ----
N = 1  # change me
P_unit, edges = subdivided_icosahedron_points_edges(N)
plot_icosphere(P_unit, edges, show_points=True, point_size=6)
