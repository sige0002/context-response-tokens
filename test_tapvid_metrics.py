import numpy as np

from tapvid_metrics import compute_tapvid_metrics, sample_queries_first


def test_first_visible_queries_occlusion_and_strict_pixel_thresholds():
    points = np.zeros((1, 4, 2), dtype=np.float32)
    occluded = np.array([[True, False, False, True]])
    sampled = sample_queries_first(occluded, points, np.zeros((4, 2, 2, 3), dtype=np.uint8))
    np.testing.assert_array_equal(sampled["query_points"], [[[1, 0, 0]]])
    predicted = sampled["target_points"].copy()
    predicted[..., 0] += 1  # Exactly one pixel: fails strict <1, passes <2.
    predicted[:, :, :2] = 100  # Before/query-frame errors must be excluded.
    metrics = compute_tapvid_metrics(sampled["query_points"], sampled["occluded"],
                                     sampled["target_points"], np.zeros((1, 1, 4), dtype=bool),
                                     predicted, query_mode="first")
    np.testing.assert_allclose(metrics["pts_within_1"], [0])
    np.testing.assert_allclose(metrics["pts_within_2"], [1])
    np.testing.assert_allclose(metrics["average_pts_within_thresh"], [.8])
    np.testing.assert_allclose(metrics["occlusion_accuracy"], [.5])
    np.testing.assert_allclose(metrics["jaccard_2"], [.5])
    np.testing.assert_allclose(metrics["average_jaccard"], [.4])
