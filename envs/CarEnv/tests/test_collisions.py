import numpy as np
from CarEnv.Collision import intersection_distances_aabb_circles


def test_intersection_distances_aabb_circles_within():
    aabb = (-2., 3., 1., 5.)
    r = 1.5

    centers = np.array([
        (-1., 3.),  # Fully within AABB -> 0.
        (-2.8, 2.),   # Below min x edge -> .8
        (3.7, 5.),   # Above max x edge -> .7
        (-1., .4),   # Below min y edge -> .6
        (0., 5.3),   # Above max y edge -> .3
        (-2.5, 0.5),  # Lower left corner -> sqrt(.5)
        (3.5, 0.5),  # Lower right corner -> sqrt(.5)
        (3.5, 5.5),  # Upper right corner -> sqrt(.5)
        (-2.5, 5.5),  # Upper left corner -> sqrt(.5)
    ])

    expected_distances = np.array([0., .8, .7, .6, .3, np.sqrt(.5), np.sqrt(.5), np.sqrt(.5), np.sqrt(.5)])

    distances = intersection_distances_aabb_circles(aabb, r, centers)

    np.testing.assert_allclose(expected_distances, distances)


def test_intersection_distances_aabb_circles_outside():
    aabb = (-2., 3., 1., 5.)
    r = 1.5

    centers = np.array([
        (-3.5, -.5),
        (-3.5, 6.5),
        (4.5, 6.5),
        (4.5, -.5),
    ])

    distances = intersection_distances_aabb_circles(aabb, r, centers)

    np.testing.assert_array_less(np.full_like(distances, r), distances)
