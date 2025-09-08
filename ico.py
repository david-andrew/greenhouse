import numpy as np
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def _rotation_matrix_x(degrees):
    rad = np.deg2rad(degrees)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def _rotation_matrix_from_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float).reshape(3,)
    b = np.asarray(b, dtype=float).reshape(3,)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return np.eye(3)
    a = a / na; b = b / nb
    c = float(np.dot(a, b))
    if c > 0.999999:
        return np.eye(3)
    if c < -0.999999:
        # 180-degree rotation: choose any axis orthogonal to a
        ref = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, ref)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm == 0.0:
            return -np.eye(3)
        axis = axis / axis_norm
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=float)
        # theta = pi => sin=0, 1-cos=2
        return np.eye(3) + 2.0 * (K @ K)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=float)
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))


def icosahedron(rotate_x_degrees=0.0):
    """Return (V, F): 12 vertices and 20 triangular faces (indexing into V).

    rotate_x_degrees: optional rotation about X axis applied to vertices.
    """
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
    if rotate_x_degrees:
        R = _rotation_matrix_x(rotate_x_degrees)
        V = V @ R.T
    return V, F

def set_axes_equal(ax):
    x_limits, y_limits, z_limits = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    x_range, y_range, z_range = abs(x_limits[1]-x_limits[0]), abs(y_limits[1]-y_limits[0]), abs(z_limits[1]-z_limits[0])
    x_mid, y_mid, z_mid = np.mean(x_limits), np.mean(y_limits), np.mean(z_limits)
    r = 0.5*max([x_range, y_range, z_range])
    ax.set_xlim3d([x_mid-r, x_mid+r]); ax.set_ylim3d([y_mid-r, y_mid+r]); ax.set_zlim3d([z_mid-r, z_mid+r])

def subdivided_icosahedron_points_edges(N, rotate_x_degrees=0.0):
    """
    N = number of interior points inserted along each original edge
        (i.e., edge is split into N+1 segments).
    Returns:
      P_unit: (V,3) points on the unit sphere
      edges:  list of (i,j) index pairs into P_unit
      rotate_x_degrees: optional rotation about X axis applied to base icosahedron
    """
    V, F = icosahedron(rotate_x_degrees=rotate_x_degrees)
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


def subdivided_icosahedron_geometry(N, rotate_x_degrees=0.0):
    """
    Same as subdivided_icosahedron_points_edges, but also returns triangular faces.
    Returns (P_unit, edges, faces) where faces are (i,j,k) CCW w.r.t. the base triangle.
    """
    V, F = icosahedron(rotate_x_degrees=rotate_x_degrees)
    f = N + 1
    point_index, points, edges_set = {}, [], set()
    faces: List[Tuple[int, int, int]] = []

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

        # Triangular tiling edges and faces
        for a in range(f):
            for b in range(f - a):
                A = local[(a, b)]
                B = local[(a+1, b)]
                C = local[(a, b+1)]
                # small upright triangle (A,B,C)
                faces.append((A, B, C))
                for u, v in ((A,B),(A,C),(B,C)): edges_set.add(tuple(sorted((u,v))))
                if a + b < f - 1:
                    D = local[(a+1, b+1)]
                    # small inverted triangle (B,D,C)
                    faces.append((B, D, C))
                    for u, v in ((B,D),(C,D),(B,C)): edges_set.add(tuple(sorted((u,v))))

    P = np.asarray(points, float)
    norms = np.linalg.norm(P, axis=1, keepdims=True); norms[norms==0.0] = 1.0
    P_unit = P / norms
    return P_unit, sorted(edges_set), faces


@dataclass
class Mesh:
    points: np.ndarray  # shape (V,3)
    edges: List[Tuple[int, int]]
    faces: Optional[List[Tuple[int, int, int]]] = None


def mesh_filtered_by_point_mask(mesh: Mesh, keep_mask: np.ndarray) -> Mesh:
    if keep_mask.dtype != bool:
        raise ValueError("keep_mask must be a boolean array")
    if keep_mask.shape[0] != mesh.points.shape[0]:
        raise ValueError("keep_mask length must match number of points")
    new_index = -np.ones(mesh.points.shape[0], dtype=int)
    new_index[keep_mask] = np.arange(int(np.count_nonzero(keep_mask)))
    new_points = mesh.points[keep_mask]
    new_edges: List[Tuple[int, int]] = []
    for i, j in mesh.edges:
        ni, nj = int(new_index[i]), int(new_index[j])
        if ni >= 0 and nj >= 0:
            new_edges.append((ni, nj))
    new_faces: Optional[List[Tuple[int, int, int]]] = None
    if mesh.faces is not None:
        new_faces = []
        for a, b, c in mesh.faces:
            na, nb, nc = int(new_index[a]), int(new_index[b]), int(new_index[c])
            if na >= 0 and nb >= 0 and nc >= 0:
                new_faces.append((na, nb, nc))
    return Mesh(new_points, new_edges, new_faces)


def mesh_filtered_by_predicate(mesh: Mesh, predicate: Callable[[np.ndarray], np.ndarray]) -> Mesh:
    mask = predicate(mesh.points)
    return mesh_filtered_by_point_mask(mesh, mask)


def mesh_keep_points_with_z_between(mesh: Mesh, min_z: Optional[float] = None, max_z: Optional[float] = None) -> Mesh:
    z = mesh.points[:, 2]
    mask = np.ones_like(z, dtype=bool)
    if min_z is not None:
        mask &= (z >= min_z)
    if max_z is not None:
        mask &= (z <= max_z)
    return mesh_filtered_by_point_mask(mesh, mask)


def mesh_scaled(mesh: Mesh, scale: float, center: Optional[np.ndarray] = None) -> Mesh:
    """Return a new Mesh with points scaled uniformly about center (default origin)."""
    if center is None:
        center = np.zeros(3, dtype=float)
    center = np.asarray(center, dtype=float).reshape(3,)
    new_points = (mesh.points - center) * float(scale) + center
    new_edges = list(mesh.edges)
    new_faces = list(mesh.faces) if mesh.faces is not None else None
    return Mesh(new_points, new_edges, new_faces)


def compute_face_lengths_and_angles(points: np.ndarray, faces: List[Tuple[int, int, int]]):
    """For each face (i,j,k), compute edge lengths (|AB|,|BC|,|CA|) and internal angles at A,B,C in degrees.

    Returns list of tuples: (lengths_tuple, angles_tuple)
    """
    results = []
    for (i, j, k) in faces:
        A, B, C = points[i], points[j], points[k]
        AB = B - A; BC = C - B; CA = A - C
        BA = -AB; CB = -BC; AC = -CA
        len_AB = float(np.linalg.norm(AB))
        len_BC = float(np.linalg.norm(BC))
        len_CA = float(np.linalg.norm(CA))

        def angle_between(u: np.ndarray, v: np.ndarray) -> float:
            nu = float(np.linalg.norm(u)); nv = float(np.linalg.norm(v))
            if nu == 0.0 or nv == 0.0:
                return 0.0
            cosang = float(np.dot(u, v) / (nu * nv))
            cosang = max(-1.0, min(1.0, cosang))
            return float(np.degrees(np.arccos(cosang)))

        angle_A = angle_between(AB, AC)
        angle_B = angle_between(BA, BC)
        angle_C = angle_between(CA, CB)

        results.append(((len_AB, len_BC, len_CA), (angle_A, angle_B, angle_C)))
    return results


def print_mesh_face_metrics(mesh: Mesh, decimals: int = 3) -> None:
    if not mesh.faces:
        print("No faces to analyze.")
        return
    metrics = compute_face_lengths_and_angles(mesh.points, mesh.faces)
    fmt = f"{{:.{decimals}f}}"
    for idx, ((l_ab, l_bc, l_ca), (ang_a, ang_b, ang_c)) in enumerate(metrics):
        lengths_str = f"(AB={fmt.format(l_ab)}, BC={fmt.format(l_bc)}, CA={fmt.format(l_ca)})"
        angles_str = f"(A={fmt.format(ang_a)}°, B={fmt.format(ang_b)}°, C={fmt.format(ang_c)}°)"
        i, j, k = mesh.faces[idx]
        print(f"Face {idx:03d} verts=({i},{j},{k}) lengths={lengths_str} angles={angles_str}")


def compute_joint_unit_vectors(points: np.ndarray, edges: List[Tuple[int, int]], align_to_z: bool = False):
    """Return list where entry i is a list of unit vectors along each incident edge at vertex i.

    Each vector points outward from the joint (vertex i) toward its neighbor, as if the joint
    were translated to the origin (0,0,0).
    """
    num_points = points.shape[0]
    neighbors: List[List[int]] = [[] for _ in range(num_points)]
    for i, j in edges:
        neighbors[i].append(j)
        neighbors[j].append(i)
    joint_vectors: List[List[np.ndarray]] = []
    for i in range(num_points):
        origin = points[i]
        R = None
        if align_to_z:
            R = _rotation_matrix_from_a_to_b(origin, np.array([0.0, 0.0, 1.0], dtype=float))
        vecs: List[np.ndarray] = []
        for j in neighbors[i]:
            v = points[j] - origin
            n = float(np.linalg.norm(v))
            if n == 0.0:
                continue
            u = v / n
            if R is not None:
                u = R @ u
            vecs.append(u)
        joint_vectors.append(vecs)
    return joint_vectors


def print_joint_unit_vectors(mesh: Mesh, decimals: int = 3, align_to_z: bool = False) -> None:
    vecs_per_joint = compute_joint_unit_vectors(mesh.points, mesh.edges, align_to_z=align_to_z)
    fmt = f"{{:.{decimals}f}}"
    for i, vecs in enumerate(vecs_per_joint):
        print(f"Joint {i:03d} ({len(vecs)} edges){' [aligned to +Z]' if align_to_z else ''}:")
        for k, v in enumerate(vecs):
            print(f"  u{k}: ({fmt.format(v[0])}, {fmt.format(v[1])}, {fmt.format(v[2])})")


def subdivided_icosahedron_mesh(N: int, rotate_x_degrees: float = 0.0) -> Mesh:
    P_unit, edges, faces = subdivided_icosahedron_geometry(N, rotate_x_degrees=rotate_x_degrees)
    return Mesh(P_unit, edges, faces)

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


def plot_mesh(mesh: Mesh, show_points: bool = True, point_size: int = 100, show_faces: bool = False, face_color: str = 'tab:blue', face_alpha: float = 0.15):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect((1,1,1))
    if show_points:
        ax.scatter(mesh.points[:,0], mesh.points[:,1], mesh.points[:,2], s=50.0, color='r')
    if show_faces and mesh.faces:
        triangles = [mesh.points[[a, b, c]] for (a, b, c) in mesh.faces]
        poly = Poly3DCollection(triangles, facecolors=face_color, edgecolors='none', alpha=face_alpha)
        ax.add_collection3d(poly)
    for i, j in mesh.edges:
        ax.plot([mesh.points[i,0],mesh.points[j,0]], [mesh.points[i,1],mesh.points[j,1]], [mesh.points[i,2],mesh.points[j,2]], color='k', linewidth=1.2)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    set_axes_equal(ax)
    ax.set_title('Mesh')
    plt.show()


def plot_joint_vectors(mesh: Mesh, joint_index: int, align_to_z: bool = False, scale: float = 1.0):
    """Visualize one joint: its vectors and the alignment.

    - If align_to_z, rotates the vectors so the joint normal (position) points to +Z.
    - Plots vectors as arrows from the origin in the rotated frame.
    """
    vecs_per_joint = compute_joint_unit_vectors(mesh.points, mesh.edges, align_to_z=align_to_z)
    if joint_index < 0 or joint_index >= len(vecs_per_joint):
        raise IndexError("joint_index out of range")
    vecs = vecs_per_joint[joint_index]

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect((1,1,1))
    # draw unit sphere grid for reference
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color='lightgray', linewidth=0.3, rstride=4, cstride=4)
    # plot vectors
    for q in vecs:
        ax.plot([0, q[0]*scale], [0, q[1]*scale], [0, q[2]*scale], color='k', linewidth=2.0)
    ax.scatter([0], [0], [0], color='r', s=30)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    set_axes_equal(ax)
    ax.set_title(f'Joint {joint_index} vectors' + (' (aligned +Z)' if align_to_z else ''))
    plt.show()


def plot_all_joints_vectors(mesh: Mesh, align_to_z: bool = False, scale: float = 1.0, indices: Optional[List[int]] = None):
    """Plot joint vectors for each joint in indices (or all joints if None)."""
    if indices is None:
        indices = list(range(mesh.points.shape[0]))
    for idx in indices:
        plot_joint_vectors(mesh, joint_index=idx, align_to_z=align_to_z, scale=scale)

# ---- Example run ----
if __name__ == "__main__":
    N = 1  # change me
    diameter = 28 * 12 # inches (TODO: model in mm)
    # Rotate the base icosahedron and build Mesh
    mesh = subdivided_icosahedron_mesh(N, rotate_x_degrees=30)
    # Example: keep only points with z >= -0.1
    mesh = mesh_keep_points_with_z_between(mesh, min_z=-0.1)
    # Scale outward by 1.5x
    mesh = mesh_scaled(mesh, scale=diameter / 2)
    plot_mesh(mesh, show_points=True, point_size=6, show_faces=True, face_color='tab:blue', face_alpha=0.2)
    # Print per-face edge lengths and internal angles
    print_mesh_face_metrics(mesh, decimals=3)
    # Print unit vectors for each joint (vertex)
    print_joint_unit_vectors(mesh, decimals=3, align_to_z=True)
    # Visualize a single joint's rotated vectors
    plot_joint_vectors(mesh, joint_index=0, align_to_z=True, scale=diameter*0.05)
    # Or visualize all joints (this will open many figures)
    # plot_all_joints_vectors(mesh, align_to_z=True, scale=diameter*0.05)
