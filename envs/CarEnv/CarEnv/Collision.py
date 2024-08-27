import numpy as np


def intersections_lines_circles(origins, ray_normals, r, centers):
    assert len(ray_normals.shape) == 2
    assert len(centers.shape) == 2
    assert len(origins) == len(ray_normals)
    assert origins.shape[-1] == ray_normals.shape[-1]
    assert centers.shape[-1] == ray_normals.shape[-1]
    assert np.asarray(r).shape == tuple()

    if len(centers) == 0:
        return np.zeros(len(origins)), np.full(len(origins), -1)

    # Find scalar projections between all lines and all circle centers
    a = ray_normals[None, :, None, :] @ centers[:, None, :, None]
    a = np.squeeze(a, (-1, -2))

    # Find the point that is closest on the ray
    close_points = origins + ray_normals * a[..., None]

    # Calculate squared distance of that point
    dist_sq = np.sum(np.square(close_points - centers[:, None]), axis=-1)

    # Only consider if close enough and in front of origin, the corner cases of intersecting back through the origin
    # is ignored
    valid = (dist_sq < r**2) & (a > 0.0)

    # For all valid intersects calculate the actual scalar intersection, this can become less than zero if the circle
    # clips, consider this as zero
    true_dist = a[valid] - np.maximum(0.0, np.sqrt(r**2 - dist_sq[valid]))

    result = np.full_like(dist_sq, np.inf)
    result[valid] = true_dist

    best = np.argmin(result, 0)
    assert best.shape == (len(ray_normals),)

    js = np.arange(len(ray_normals))
    result_distance = result[best, js]
    result_idxs = np.where(np.isfinite(result_distance), best, -1)
    result_distance = np.where(np.isfinite(result_distance), result_distance, 0.0)

    return result_distance, result_idxs


def intersections_aabb_circles(aabb, r, centers):
    assert np.asarray(aabb).shape == (4,)
    assert np.asarray(r).shape == ()
    assert len(centers.shape) == 2
    assert centers.shape[1] == 2

    x1, x2, y1, y2 = aabb

    candidate_mask = np.logical_and(
        np.logical_and(x1 - r < centers[:, 0], centers[:, 0] < x2 + r),
        np.logical_and(y1 - r < centers[:, 1], centers[:, 1] < y2 + r),
    )

    candidates = centers[candidate_mask]

    candidate_simple_match = np.logical_or(
        np.logical_and(
            np.logical_and(x1 < candidates[:, 0], candidates[:, 0] < x2),
            np.logical_and(y1 - r < candidates[:, 1], candidates[:, 1] < y2 + r),
        ),
        np.logical_and(
            np.logical_and(x1 - r < candidates[:, 0], candidates[:, 0] < x2 + r),
            np.logical_and(y1 < candidates[:, 1], candidates[:, 1] < y2),
        ),
    )

    # Check those without simple region match for proximity to corners to check if on rounded corners
    # TODO: Could only calculate for non-simple matches
    candidate_corner_match = np.logical_or(
        np.logical_or(
            (x1 - candidates[:, 0]) ** 2 + (y1 - candidates[:, 1]) ** 2 < r**2,
            (x1 - candidates[:, 0]) ** 2 + (y2 - candidates[:, 1]) ** 2 < r**2,
        ),
        np.logical_or(
            (x2 - candidates[:, 0]) ** 2 + (y1 - candidates[:, 1]) ** 2 < r**2,
            (x2 - candidates[:, 0]) ** 2 + (y2 - candidates[:, 1]) ** 2 < r**2,
        ),
    )

    result = np.zeros(len(centers), dtype=bool)
    result[candidate_mask] = np.logical_or(
        candidate_simple_match, candidate_corner_match
    )
    return result


def intersection_distances_aabb_circles(aabb, r, centers):
    assert np.asarray(aabb).shape == (4,)
    assert np.asarray(r).shape == ()
    assert len(centers.shape) == 2
    assert centers.shape[1] == 2

    x1, x2, y1, y2 = aabb

    # Intersection candidates identical to intersection case
    candidate_mask = np.logical_and(
        np.logical_and(x1 - r < centers[:, 0], centers[:, 0] < x2 + r),
        np.logical_and(y1 - r < centers[:, 1], centers[:, 1] < y2 + r),
    )

    candidates = centers[candidate_mask]
    candidates_dist = np.full(len(candidates), np.inf)

    x_bounds_mask = np.logical_and(x1 < candidates[:, 0], candidates[:, 0] < x2)
    y_bounds_mask = np.logical_and(y1 < candidates[:, 1], candidates[:, 1] < y2)

    candidates_dist[x_bounds_mask & y_bounds_mask] = 0.0

    # Distance to edges
    dist = np.abs(candidates[x_bounds_mask, 1] - y1)
    candidates_dist[x_bounds_mask] = np.minimum(candidates_dist[x_bounds_mask], dist)
    dist = np.abs(candidates[x_bounds_mask, 1] - y2)
    candidates_dist[x_bounds_mask] = np.minimum(candidates_dist[x_bounds_mask], dist)
    dist = np.abs(candidates[y_bounds_mask, 0] - x1)
    candidates_dist[y_bounds_mask] = np.minimum(candidates_dist[y_bounds_mask], dist)
    dist = np.abs(candidates[y_bounds_mask, 0] - x2)
    candidates_dist[y_bounds_mask] = np.minimum(candidates_dist[y_bounds_mask], dist)

    # Distance to corners
    for corner in np.array([(x1, y1), (x1, y2), (x2, y1), (x2, y2)]):
        dists = np.linalg.norm(candidates - corner, axis=-1)
        candidates_dist = np.minimum(candidates_dist, dists)

    result = np.full(len(centers), np.inf)
    result[candidate_mask] = candidates_dist
    return result


def intersection_aabb_lines(aabb, r, lines):
    """Calculates which lines intersect with the AABB

    Args:
        aabb (Tuple): Tuple describing the axis aligned bound box (x1, x2, y1, y2)
        r (float): _description_
        lines (np.ndarray): Nx4 array of lines with each row describing one line (x_start, y_start, x_end, y_end)

    Returns:
        np.ndarray: Nx1 boolean array indicating whether the corresponding line intersected with the aabb
    """
    assert np.asarray(aabb).shape == (4,)
    assert np.asarray(r).shape == ()
    assert len(lines.shape) == 2
    assert lines.shape[1] == 4

    x1, x2, y1, y2 = aabb
    # lines = (x_start, y_start, x_end, y_end)
    start = lines[:, :2]
    end = lines[:, 2:]

    # check if the lines are completely to the left, right, above, or below the AABB.
    outside_x = np.logical_or(
        np.logical_and(start[:, 0] < x1, end[:, 0] < x1),
        np.logical_and(start[:, 0] > x2, end[:, 0] > x2),
    )
    outside_y = np.logical_or(
        np.logical_and(start[:, 1] < y1, end[:, 1] < y1),
        np.logical_and(start[:, 1] > y2, end[:, 1] > y2),
    )

    # calculate the direction vectors of the lines
    dir_x = end[:, 0] - start[:, 0]
    dir_y = end[:, 1] - start[:, 1]

    # handle case where car might be perfectly parallel to some lines
    eps = 1e-6
    dir_x = np.where(dir_x == 0, eps, dir_x)
    dir_y = np.where(dir_y == 0, eps, dir_y)

    # calculate parameterized values of intersection for each side of the AABB
    t_left = (x1 - start[:, 0]) / dir_x
    t_right = (x2 - start[:, 0]) / dir_x
    t_top = (y1 - start[:, 1]) / dir_y
    t_bottom = (y2 - start[:, 1]) / dir_y

    # determine the range of t-values where the line intersects the AABB
    entry = np.maximum(np.minimum(t_left, t_right), np.minimum(t_top, t_bottom))
    exit = np.minimum(np.maximum(t_left, t_right), np.maximum(t_top, t_bottom))
    # not overlapping -> no intersection
    not_overlapping = entry > exit

    # check if the intersection points are within the line segments
    valid_intersection = np.logical_and(entry >= 0, entry <= 1)

    intersects = (
        np.logical_not(np.logical_or(outside_x, outside_y) | not_overlapping)
        & valid_intersection
    )

    return intersects


def intersection_distance_aabb_lines(aabb, r, lines):
    """Calculates the distances at which lines intersect with the AABB

    Args:
        aabb (Tuple): Tuple describing the axis aligned bound box (x1, x2, y1, y2)
        r (float): _description_
        lines (np.ndarray): Nx4 array of lines with each row describing one line (x_start, y_start, x_end, y_end)

    Returns:
        np.ndarray: Nx1 float array with the distance to intersection for each line (np.inf for lines with no intersection)
    """
    assert np.asarray(aabb).shape == (4,)
    assert np.asarray(r).shape == ()
    assert len(lines.shape) == 2
    assert lines.shape[1] == 4

    x1, x2, y1, y2 = aabb
    # lines = (x_start, y_start, x_end, y_end)
    start = lines[:, :2]
    end = lines[:, 2:]

    # check if the lines are completely to the left, right, above, or below the AABB.
    outside_x = np.logical_or(
        np.logical_and(start[:, 0] < x1, end[:, 0] < x1),
        np.logical_and(start[:, 0] > x2, end[:, 0] > x2),
    )
    outside_y = np.logical_or(
        np.logical_and(start[:, 1] < y1, end[:, 1] < y1),
        np.logical_and(start[:, 1] > y2, end[:, 1] > y2),
    )

    # calculate the direction vectors of the lines
    dir_x = end[:, 0] - start[:, 0]
    dir_y = end[:, 1] - start[:, 1]

    # handle case where car might be perfectly parallel to some lines
    eps = 1e-6
    dir_x = np.where(dir_x == 0, eps, dir_x)
    dir_y = np.where(dir_y == 0, eps, dir_y)

    # calculate parameterized values of intersection for each side of the AABB
    t_left = (x1 - start[:, 0]) / dir_x
    t_right = (x2 - start[:, 0]) / dir_x
    t_top = (y1 - start[:, 1]) / dir_y
    t_bottom = (y2 - start[:, 1]) / dir_y

    # determine the range of t-values where the line intersects the AABB
    entry = np.maximum(np.minimum(t_left, t_right), np.minimum(t_top, t_bottom))
    exit = np.minimum(np.maximum(t_left, t_right), np.maximum(t_top, t_bottom))

    # calculate intersection points
    intersection_points_x = start[:, 0] + entry * dir_x
    intersection_points_y = start[:, 1] + entry * dir_y

    # calculate distances
    distances = np.sqrt(
        (start[:, 0] - intersection_points_x) ** 2
        + (start[:, 1] - intersection_points_y) ** 2
    )

    # not overlapping -> no intersection
    not_overlapping = entry > exit

    # check if the intersection points are within the line segments
    valid_intersection = np.logical_and(entry >= 0, entry <= 1)

    intersects = (
        np.logical_not(np.logical_or(outside_x, outside_y) | not_overlapping)
        & valid_intersection
    )

    return np.where(intersects, distances, np.inf)
