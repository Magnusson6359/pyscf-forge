import numpy as np
from pyscf import gto
from scipy.spatial import HalfspaceIntersection, ConvexHull, QhullError
from scipy.optimize import linprog

def get_cavity_volume(mol, radii_table):
    """
    Exact analytical volume of the van der Waals cavity using a decomposition
    via power diagrams. Based on (but not exactly following) the approach of 
    Cazals et al., Computing the volume of a union of balls: A certified algorithm.
    ACM Trans. Math. Softw. 38, 3:1-3:20 (2011).

    Parameters
    ----------
    mol : gto.Mole
        PySCF molecule object.
    radii_table : array
        Table of van der Waals radii for each atomic species.
        
    Returns
    -------
    float
        Cavity volume in Bohr^3.
    """
    coords = mol.atom_coords()
    radii = _get_cavity_radii(mol, radii_table=radii_table)
    radii = np.array(radii)
    
    total_volume = 0.0
    
    for i in range(len(coords)): # For each atom i:
        # Get radical planes between atom i and all other atoms
        rad_planes = _get_radical_planes(coords, radii, i)
        
        # Add bounding box around atom i to ensure a closed region for HalfspaceIntersection.
        c_i = coords[i]
        r_i = radii[i]
        bbox_planes = np.array([
            [-1.0,  0.0,  0.0, c_i[0] - r_i],
            [ 1.0,  0.0,  0.0, -(c_i[0] + r_i)],
            [ 0.0, -1.0,  0.0, c_i[1] - r_i],
            [ 0.0,  1.0,  0.0, -(c_i[1] + r_i)],
            [ 0.0,  0.0, -1.0, c_i[2] - r_i],
            [ 0.0,  0.0,  1.0, -(c_i[2] + r_i)],
        ])
        if len(rad_planes) > 0:
            all_halfspaces = np.vstack([rad_planes, bbox_planes])
        else:
            all_halfspaces = bbox_planes
            
        # Find an interior point of C_i using scipy.linprog
        norm_A = np.linalg.norm(all_halfspaces[:, :3], axis=1)
        A_lp = np.hstack([all_halfspaces[:, :3], norm_A[:, None]])
        b_lp = -all_halfspaces[:, 3]
        c_lp = np.array([0, 0, 0, -1])  # Maximise alpha => Minimise -alpha
        bounds_lp = [(None, None), (None, None), (None, None), (None, None)]
        res = linprog(c_lp, A_ub=A_lp, b_ub=b_lp, bounds=bounds_lp)
        if not res.success or res.x[3] <= 0:
            # Atom i is completely swallowed by others, C_i is empty.
            continue
        interior_point = res.x[:3]
        
        # Get C_i (the convex polyhedron for atom i)
        try: # Catch Qhull errors if C_i degenerate
            convex_polyhedron = HalfspaceIntersection(all_halfspaces, interior_point)
            
            # Extract the planar faces of C_i
            planar_faces = _extract_planar_faces(convex_polyhedron)
            c_i = coords[i]
            r_i = radii[i]
            
            # Compute planar face volume contribution (and solid angle)
            C_i_volume = 0.0
            omega_P_total = 0.0
            for face in planar_faces:
                n = face['normal']
                d = face['d']
                verts = face['vertices']
                h_dist = np.abs(np.dot(n, c_i) + d)
                if h_dist >= r_i:
                    continue # This face is outside the sphere, no contribution

                R_circle = np.sqrt(r_i**2 - h_dist**2) # Radius of the boundary circle for the planar face
                signed_h = np.dot(n, c_i) + d # Positive if center is on the same side as normal, negative otherwise
                p0 = c_i - signed_h * n # Origin for 2d projection

                # Project the vertices of the face to 2D coordinates in the plane of the face
                u, v = _get_orthogonal_basis_for_plane(n)
                verts_2d = np.vstack([np.dot(verts - p0, u), np.dot(verts - p0, v)]).T

                # Compute the area and solid angle of the intersection of the planar face with its boundary circle
                area_P, omega_P = _polygon_circle_intersection_area_and_solid_angle(verts_2d, R_circle, -signed_h, r_i)
                omega_P_total += omega_P
                
                # Volume contribution for the planar faces (generalised pyramid formula)
                C_i_volume += (-signed_h * area_P) / 3.0
                
            # Spherical cap contribution
            # Motivation: The total solid angle of C_i is 4pi, so 4pi-omega_P_total is the solid angle of the (total) spherical cap.
            # "Everything not a planar face is a spherical cap"
            is_center_inside = np.all(all_halfspaces[:, :3] @ c_i + all_halfspaces[:, 3] <= 1e-8)
            omega_S = (4.0 * np.pi if is_center_inside else 0.0) - omega_P_total
            area_S_exact = (r_i**2) * omega_S
            # Volume contribution from the spherical cap (generalised pyramid formula)
            C_i_volume += (r_i * area_S_exact) / 3.0
            # Total contribution from atom i:
            total_volume += C_i_volume
            
        except QhullError as e:
            # Almost empty or degenerate cell, skip this atom
            print(f"Warning: HalfspaceIntersection failed for atom {i}: {e}")
            continue

    return total_volume

def _get_cavity_radii(mol, radii_table):
    """
    Resolve cavity radii in Bohr as a per-atom array.
    """
    return np.array([
        radii_table[gto.charge(mol.atom_symbol(i))]
        for i in range(mol.natm)
    ], dtype=float)

def _solid_angle_triangle(R1, R2, R3):
    """
    Computes the solid angle subtended by the triangle formed by R1, R2, R3 at the origin.
    From Van Oosterom and Strackee, "The Solid Angle of a Plane Triangle", IEEE Transactions on Biomedical Engineering, 1983.
    """
    r1 = np.linalg.norm(R1)
    r2 = np.linalg.norm(R2)
    r3 = np.linalg.norm(R3)
    num = np.dot(R1, np.cross(R2, R3))
    den = r1*r2*r3 + np.dot(R1, R2)*r3 + np.dot(R2, R3)*r1 + np.dot(R3, R1)*r2
    return 2 * np.arctan2(num, den)

def _segment_circle_intersection_area_and_solid_angle(A, B, R, h, r_i):
    """
    Computes both the area and the exact solid angle of the intersection of 
    a triangle (Origin, A, B) and a circle of radius R centered at the origin.
    """
    # Check if A and B are collinear with the origin (degenerate triangle)
    if np.isclose(A[0]*B[1] - A[1]*B[0], 0):
        return 0.0, 0.0

    # Set up the quadratic for line-circle intersection  
    D = B - A
    a_coef = np.dot(D, D)
    b_coef = 2 * np.dot(A, D)
    c_coef = np.dot(A, A) - R**2
    
    # discriminant for intersection
    disc = b_coef**2 - 4*a_coef*c_coef
    
    # Solve for t values of intersection points, only consider those strictly between A and B (0 < t < 1)
    t_roots = []
    if disc > 0 and not np.isclose(a_coef, 0):
        root = np.sqrt(disc)
        t1 = (-b_coef - root) / (2*a_coef)
        t2 = (-b_coef + root) / (2*a_coef)
        t_roots.extend([t for t in (t1, t2) if 0 < t < 1])
    t_roots = sorted(t_roots)
    pts = [A] + [A + t*D for t in t_roots] + [B] # Points along the segment in order
    
    area = 0.0
    omega = 0.0
    for i in range(len(pts) - 1): # For each segment between consecutive points
        p1 = pts[i]
        p2 = pts[i+1]
        
        mid = (p1 + p2) / 2
        
        if np.linalg.norm(mid) <= R + 1e-8:
            # Segment is inside the circle: triangle area via shoelace formula
            area += 0.5 * (p1[0]*p2[1] - p1[1]*p2[0])
            R1 = np.array([0, 0, h])
            R2 = np.array([p1[0], p1[1], h])
            R3 = np.array([p2[0], p2[1], h])
            omega += _solid_angle_triangle(R1, R2, R3)
        else:
            # Segment is outside the circle: circular sector area
            theta1 = np.arctan2(p1[1], p1[0])
            theta2 = np.arctan2(p2[1], p2[0])
            d_theta = (theta2 - theta1 + np.pi) % (2*np.pi) - np.pi
            area += 0.5 * R**2 * d_theta
            if not np.isclose(h, 0):
                omega += d_theta * (np.sign(h) - h / r_i)
            # if h == 0, omega contribution is 0.0
            
    return area, omega

def _polygon_circle_intersection_area_and_solid_angle(verts_2d, R, h, r_i):
    """
    Finds the exact intersection area and solid angle between an arbitrary 
    2D polygon and a circle of radius R centered at the origin.
    """
    area_total = 0.0
    omega_total = 0.0
    num_verts = len(verts_2d)
    for k in range(num_verts): # For each face
        A = verts_2d[k]
        B = verts_2d[(k+1) % num_verts]
        a, o = _segment_circle_intersection_area_and_solid_angle(A, B, R, h, r_i)
        area_total += a
        omega_total += o
        
    return abs(area_total), omega_total

def _extract_planar_faces(convex_polyhedrons, tol=1e-8):
    """
    Extracts the planar faces of the convex polyhedron defined by a HalfspaceIntersection.
    """
    vertices = convex_polyhedrons.intersections # vertices of the polyhedron
    halfspaces = convex_polyhedrons.halfspaces # radical planes + bounding box planes
    
    faces = []
    for h in halfspaces:
        # Check distance of all vertices to radical plane
        dists = np.dot(vertices, h[:3]) + h[3]
        
        # Vertices lying on the plane
        on_plane_idx = np.where(np.abs(dists) <= tol)[0]
        
        if len(on_plane_idx) >= 3: # At least 3 vertices needed to define a face
            face_verts = vertices[on_plane_idx]
            
            # Order the vertices in the plane of the face by projecting to 2D and using Convex Hull
            normal = h[:3] / np.linalg.norm(h[:3])
            u, v = _get_orthogonal_basis_for_plane(normal)

            # Project vertices
            proj_2d = np.vstack([np.dot(face_verts, u), np.dot(face_verts, v)]).T

            # Sort via 2D Convex Hull
            hull_2d = ConvexHull(proj_2d)
            ordered_verts = face_verts[hull_2d.vertices]
            
            faces.append({
                'normal': normal,
                'd': h[3] / np.linalg.norm(h[:3]),
                'vertices': ordered_verts
            })
            
    return faces

def _get_radical_planes(coords, radii, i):
    """
    Construct the radical planes for the i-th sphere
    against all other spheres j in a power diagram.
    
    For two spheres i and j, the power distances to a point x are equal when:
    ||x - c_i||^2 - r_i^2 = ||x - c_j||^2 - r_j^2
    This simplifies to a plane equation: 2(c_j - c_i) * x + (||c_i||^2 - ||c_j||^2 - r_i^2 + r_j^2) = 0
    """
    N = len(coords)
    c_i = coords[i]
    r_i = radii[i]
    
    rad_planes = []
    
    for j in range(N):
        if i == j:
            continue
            
        c_j = coords[j]
        r_j = radii[j]
        
        # Power Diagram Plane: 2*(c_j - c_i)*x + (||c_i||^2 - ||c_j||^2 - r_i^2 + r_j^2) = 0
        normal = 2.0 * (c_j - c_i)
        
        c_i_2 = np.dot(c_i, c_i)
        c_j_2 = np.dot(c_j, c_j)
        
        d = (c_i_2 - c_j_2) - (r_i**2 - r_j**2)
        # Format Ax + By + Cz + D <= 0 for HalfspaceIntersection  
        rad_planes.append([normal[0], normal[1], normal[2], d])
        
    return np.array(rad_planes)

def _get_orthogonal_basis_for_plane(normal):
    """
    Given a (orthonormal) normal vector, find two orthogonal vectors that form a basis for the plane.
    """
    if abs(normal[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])
        
    u = np.cross(normal, ref)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    
    return u, v