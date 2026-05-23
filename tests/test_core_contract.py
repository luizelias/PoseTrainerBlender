import math

import numpy as np
import pytest

import pose_trainer_core as core

if core.USING_FALLBACK:
    pytest.skip("compiled Eigen C++ core is not built", allow_module_level=True)


def _square_faces():
    return [[0, 1, 2, 3]]


def _square_bind():
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )


def test_simplex_projection_is_not_clamp_and_normalize():
    projected = np.array(core.project_simplex([0.8, 0.8, -0.2]), dtype=np.float32)

    assert np.all(projected >= 0.0)
    assert math.isclose(float(projected.sum()), 1.0, abs_tol=1.0e-6)
    assert np.allclose(projected, [0.5, 0.5, 0.0], atol=1.0e-6)


def test_train_rejects_sample_vertex_count_mismatch():
    settings = core.PoseTrainerSettings()
    bind = _square_bind()
    bad_sample = bind[:3]

    with pytest.raises(ValueError):
        core.train(
            _square_faces(),
            bind,
            [bad_sample],
            [{"name": "all", "weights": np.ones(len(bind), dtype=np.float32)}],
            settings,
        )


def test_envelope_zero_passthrough():
    settings = core.PoseTrainerSettings()
    settings.relax_iterations = 1
    bind = _square_bind()
    sample = bind.copy()
    sample[2, 2] = 0.5

    cache = core.train(
        _square_faces(),
        bind,
        [sample],
        [{"name": "all", "weights": np.ones(len(bind), dtype=np.float32)}],
        settings,
    )

    animated = bind.copy()
    animated[:, 2] = [0.0, 0.1, 0.2, 0.1]
    out = cache.evaluate(animated, np.ones(len(bind), dtype=np.float32), 0.0, 1)

    assert out.shape == animated.shape
    assert np.allclose(out, animated)


def test_area_relax_spreads_area_weight_in_cpp_core():
    bind = _square_bind()
    sample = bind.copy()
    sample[2, 2] = 0.5
    sparse_weights = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    sharp_settings = core.PoseTrainerSettings()
    sharp_settings.relax_iterations = 1
    sharp_settings.area_relax_iterations = 0
    sharp_cache = core.train(
        _square_faces(),
        bind,
        [sample],
        [{"name": "corner", "weights": sparse_weights}],
        sharp_settings,
    )

    blurred_settings = core.PoseTrainerSettings()
    blurred_settings.relax_iterations = 1
    blurred_settings.area_relax_iterations = 2
    blurred_cache = core.train(
        _square_faces(),
        bind,
        [sample],
        [{"name": "corner", "weights": sparse_weights}],
        blurred_settings,
    )

    sharp_out = sharp_cache.evaluate(bind, None, 1.0, 1)
    blurred_out = blurred_cache.evaluate(bind, None, 1.0, 1)

    assert not np.allclose(sharp_out, blurred_out)


def test_evaluate_returns_finite_positions():
    settings = core.PoseTrainerSettings()
    settings.relax_iterations = 1
    settings.solve_iterations = 2
    bind = _square_bind()
    sample = bind.copy()
    sample[1, 2] = 0.25
    sample[2, 2] = 0.5

    cache = core.train(
        _square_faces(),
        bind,
        [sample],
        [{"name": "all", "weights": np.ones(len(bind), dtype=np.float32)}],
        settings,
    )
    out = cache.evaluate(sample, None, 1.0, 2)

    assert out.shape == sample.shape
    assert np.isfinite(out).all()


def test_opencl_matches_cpu_on_tiny_mesh_when_available():
    if not hasattr(core, "opencl_available") or not core.opencl_available():
        pytest.skip("OpenCL runtime is not available")

    settings = core.PoseTrainerSettings()
    settings.relax_iterations = 1
    bind = _square_bind()
    sample = bind.copy()
    sample[2, 2] = 0.5

    cache = core.train(
        _square_faces(),
        bind,
        [sample],
        [{"name": "all", "weights": np.ones(len(bind), dtype=np.float32)}],
        settings,
    )

    cpu = cache.evaluate(sample, None, 1.0, 1, 1)
    opencl = cache.evaluate(sample, None, 1.0, 1, 2, True)

    assert getattr(cache, "last_backend", "") == "OpenCL"
    assert hasattr(cache, "last_opencl_timing")
    assert np.allclose(opencl, cpu, atol=1.0e-5)
