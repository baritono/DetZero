"""
Smoke tests for the DetZero framework.

These tests validate that the pure-Python portions of the codebase can be
imported and exercised in a CPU-only environment (no GPU, no compiled CUDA
extensions, no Waymo dataset files).

CUDA ops and spconv are stubbed out by conftest.py before collection begins.

Run with:
    pytest tests/smoke_test.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chdir(path):
    """Context manager: temporarily change working directory."""
    import contextlib

    @contextlib.contextmanager
    def _cm():
        old = os.getcwd()
        try:
            os.chdir(path)
            yield
        finally:
            os.chdir(old)

    return _cm()


# ===========================================================================
# 1. Import checks
# ===========================================================================


class TestImports:
    """Verify that all top-level packages and key sub-modules can be imported."""

    def test_import_config_utils(self):
        from detzero_utils import config_utils

        assert callable(config_utils.cfg_from_yaml_file)
        assert callable(config_utils.cfg_from_list)
        assert callable(config_utils.merge_new_config)

    def test_import_common_utils(self):
        from detzero_utils import common_utils

        assert callable(common_utils.create_logger)
        assert callable(common_utils.set_random_seed)
        assert callable(common_utils.get_log_info)

    def test_import_detzero_utils_structures(self):
        from detzero_utils.structures import AnnotationDict, FrameId  # noqa: F401

    def test_import_detzero_utils_box_utils(self):
        from detzero_utils import box_utils  # noqa: F401

        assert hasattr(box_utils, "boxes_to_corners_3d")

    def test_import_detzero_utils_model_utils(self):
        from detzero_utils import model_utils  # noqa: F401

    def test_import_detzero_det(self):
        import detzero_det

        assert detzero_det.__version__.startswith("0.1.0")

    def test_import_detzero_det_structures(self):
        from detzero_det.structures import (  # noqa: F401
            BatchDict,
            DataDict,
            PredictionDict,
        )

    def test_import_detzero_track(self):
        import detzero_track

        assert detzero_track.__version__.startswith("0.1.0")

    def test_import_detzero_track_structures(self):
        from detzero_track.structures import (  # noqa: F401
            FrameDetectionData,
            TrackletData,
            TrackletDataBase,
        )

    def test_import_detzero_refine(self):
        import detzero_refine

        assert detzero_refine.__version__.startswith("0.1.0")

    def test_import_centerpoint(self):
        """CenterPoint model class must be importable (CUDA ops are stubbed)."""
        from detzero_det.models.centerpoint import CenterPoint  # noqa: F401

    def test_import_refine_models(self):
        from detzero_refine.models.geometry_refine_model import (  # noqa: F401
            GeometryRefineModel,
        )
        from detzero_refine.models.position_refine_model import (  # noqa: F401
            PositionRefineModel,
        )
        from detzero_refine.models.confidence_refine_model import (  # noqa: F401
            ConfidenceRefineModel,
        )

    def test_import_iou3d_nms_utils(self):
        """iou3d_nms_utils imports the CUDA stub – must not raise."""
        from detzero_utils.ops.iou3d_nms import iou3d_nms_utils  # noqa: F401

    def test_import_det_dataset_template(self):
        """DatasetTemplate is the base class for all detection datasets."""
        from detzero_det.datasets.dataset import DatasetTemplate  # noqa: F401

    def test_import_track_dataset(self):
        pytest.importorskip("filterpy", reason="filterpy not installed (unlisted dep; needed by tracking Kalman filter)")
        from detzero_track.datasets import build_dataloader  # noqa: F401


# ===========================================================================
# 2. Config utilities
# ===========================================================================


class TestConfigUtils:
    """Test YAML config parsing and merging."""

    def test_parse_detection_dataset_config(self):
        from easydict import EasyDict

        from detzero_utils.config_utils import cfg_from_yaml_file

        cfg = EasyDict()
        cfg_path = REPO_ROOT / "detection/tools/cfgs/det_dataset_cfgs/waymo_1sweep.yaml"
        result = cfg_from_yaml_file(str(cfg_path), cfg)

        assert result["DATASET"] == "WaymoDetectionDataset"
        assert "POINT_CLOUD_RANGE" in result
        assert "DATA_SPLIT" in result
        assert len(result["POINT_CLOUD_RANGE"]) == 6

    def test_parse_detection_dataset_config_3sweep(self):
        from easydict import EasyDict

        from detzero_utils.config_utils import cfg_from_yaml_file

        cfg = EasyDict()
        cfg_path = REPO_ROOT / "detection/tools/cfgs/det_dataset_cfgs/waymo_3sweeps.yaml"
        result = cfg_from_yaml_file(str(cfg_path), cfg)

        assert result["DATASET"] == "WaymoDetectionDataset"
        assert result.get("SWEEP_COUNT") is not None

    def test_parse_tracking_dataset_config(self):
        from easydict import EasyDict

        from detzero_utils.config_utils import cfg_from_yaml_file

        cfg = EasyDict()
        cfg_path = REPO_ROOT / "tracking/tools/cfgs/tk_dataset_cfgs/waymo_dataset.yaml"
        result = cfg_from_yaml_file(str(cfg_path), cfg)

        assert result["DATASET"] == "WaymoTrackDataset"
        assert "CLASS_NAME" in result

    def test_parse_refining_dataset_config(self):
        from easydict import EasyDict

        from detzero_utils.config_utils import cfg_from_yaml_file

        cfg = EasyDict()
        cfg_path = (
            REPO_ROOT / "refining/tools/cfgs/ref_dataset_cfgs/waymo_grm_dataset.yaml"
        )
        result = cfg_from_yaml_file(str(cfg_path), cfg)

        assert result["DATASET"] == "WaymoGeometryDataset"

    def test_cfg_from_list_bool(self):
        """cfg_from_list must override a boolean key."""
        from easydict import EasyDict

        from detzero_utils.config_utils import cfg_from_list, cfg_from_yaml_file

        cfg = EasyDict()
        cfg_from_yaml_file(
            str(REPO_ROOT / "detection/tools/cfgs/det_dataset_cfgs/waymo_1sweep.yaml"),
            cfg,
        )
        assert cfg.TTA is False  # original value
        cfg_from_list(["TTA", "True"], cfg)
        assert cfg.TTA is True

    def test_cfg_from_list_wrong_key_raises(self):
        """cfg_from_list must raise AssertionError for a non-existent key."""
        from easydict import EasyDict

        from detzero_utils.config_utils import cfg_from_list, cfg_from_yaml_file

        cfg = EasyDict()
        cfg_from_yaml_file(
            str(REPO_ROOT / "detection/tools/cfgs/det_dataset_cfgs/waymo_1sweep.yaml"),
            cfg,
        )
        with pytest.raises(AssertionError):
            cfg_from_list(["NONEXISTENT_KEY", "value"], cfg)

    # ------------------------------------------------------------------
    # Known bug: _BASE_CONFIG_ paths are relative to CWD, not the file
    # ------------------------------------------------------------------

    def test_model_config_base_config_fails_from_repo_root(self):
        """
        KNOWN BUG: _BASE_CONFIG_ paths in model configs are relative to the
        working directory, not to the config file itself.

        Loading centerpoint_1sweep.yaml from the repo root fails because
        cfg_from_yaml_file opens '_BASE_CONFIG_: cfgs/det_dataset_cfgs/...'
        relative to CWD rather than relative to the config file's parent dir.
        The scripts work around this by cd-ing into detection/tools/ first.
        """
        from easydict import EasyDict

        from detzero_utils.config_utils import cfg_from_yaml_file

        cfg = EasyDict()
        cfg_path = (
            REPO_ROOT / "detection/tools/cfgs/det_model_cfgs/centerpoint_1sweep.yaml"
        )
        with pytest.raises((FileNotFoundError, OSError)):
            cfg_from_yaml_file(str(cfg_path), cfg)

    def test_model_config_base_config_succeeds_from_tools_dir(self):
        """Model configs load correctly when CWD is detection/tools/."""
        from easydict import EasyDict

        from detzero_utils.config_utils import cfg_from_yaml_file

        cfg = EasyDict()
        cfg_path = (
            REPO_ROOT / "detection/tools/cfgs/det_model_cfgs/centerpoint_1sweep.yaml"
        )
        with _chdir(REPO_ROOT / "detection/tools"):
            result = cfg_from_yaml_file(str(cfg_path), cfg)

        assert "CLASS_NAMES" in result
        assert "Vehicle" in result["CLASS_NAMES"]
        assert "MODEL" in result


# ===========================================================================
# 3. Utility functions
# ===========================================================================


class TestCommonUtils:
    """Test logging helpers and random-seed utilities."""

    def test_create_logger_no_file(self):
        from detzero_utils.common_utils import create_logger

        logger = create_logger()
        assert logger is not None
        logger.info("create_logger smoke test OK")

    def test_create_logger_with_file(self, tmp_path):
        from detzero_utils.common_utils import create_logger

        log_file = tmp_path / "smoke.log"
        logger = create_logger(log_file=str(log_file))
        logger.info("file logger smoke test")
        assert log_file.exists()
        assert "file logger smoke test" in log_file.read_text()

    def test_get_log_info_contains_text(self):
        from detzero_utils.common_utils import get_log_info

        result = get_log_info("Hello")
        assert "Hello" in result

    def test_get_log_info_default_width(self):
        from detzero_utils.common_utils import get_log_info

        result = get_log_info("test", boundary="-", total_len=100)
        # Output length should approximate 100 characters
        assert len(result) <= 100

    def test_get_log_info_long_string(self):
        """get_log_info must not truncate a string longer than total_len."""
        from detzero_utils.common_utils import get_log_info

        long_str = "x" * 120
        result = get_log_info(long_str, total_len=100)
        assert long_str in result

    def test_set_random_seed(self):
        """set_random_seed must not raise on CPU (torch.cuda calls are no-ops)."""
        from detzero_utils.common_utils import set_random_seed

        set_random_seed(42)
        # Verify numpy seed was applied
        a = np.random.randint(0, 1000, size=5)
        set_random_seed(42)
        b = np.random.randint(0, 1000, size=5)
        np.testing.assert_array_equal(a, b)

    def test_log_config_to_file(self, tmp_path):
        from easydict import EasyDict

        from detzero_utils.common_utils import create_logger
        from detzero_utils.config_utils import cfg_from_yaml_file, log_config_to_file

        cfg = EasyDict()
        cfg_from_yaml_file(
            str(REPO_ROOT / "detection/tools/cfgs/det_dataset_cfgs/waymo_1sweep.yaml"),
            cfg,
        )
        log_file = tmp_path / "cfg.log"
        logger = create_logger(log_file=str(log_file))
        log_config_to_file(cfg, logger=logger)
        content = log_file.read_text()
        assert "DATASET" in content


# ===========================================================================
# 4. Data structures
# ===========================================================================


class TestStructures:
    """Verify TypedDict structures can be instantiated and used."""

    def test_annotation_dict_basic_fields(self):
        from detzero_utils.structures import AnnotationDict

        ann = AnnotationDict(
            name=np.array(["Vehicle", "Pedestrian"]),
            score=np.array([0.9, 0.7], dtype=np.float32),
            boxes_lidar=np.zeros((2, 7), dtype=np.float32),
            sequence_name="segment_001",
            frame_id=0,
        )
        assert ann["name"].shape == (2,)
        assert ann["score"][0] == pytest.approx(0.9)

    def test_detection_typeddicts_are_dicts(self):
        from detzero_det.structures import BatchDict, DataDict, PredictionDict

        for cls in (BatchDict, DataDict, PredictionDict):
            assert issubclass(cls, dict), f"{cls.__name__} must be a dict subclass"

    def test_tracking_typeddicts_are_dicts(self):
        from detzero_track.structures import (
            FrameDetectionData,
            TrackletData,
            TrackletDataBase,
        )

        for cls in (FrameDetectionData, TrackletData, TrackletDataBase):
            assert issubclass(cls, dict), f"{cls.__name__} must be a dict subclass"

    def test_frame_detection_data_instantiation(self):
        from detzero_track.structures import FrameDetectionData

        frame = FrameDetectionData(
            name=np.array(["Vehicle"]),
            score=np.array([0.95], dtype=np.float32),
            boxes_lidar=np.zeros((1, 7), dtype=np.float32),
            boxes_global=np.zeros((1, 7), dtype=np.float32),
            sequence_name="segment_001",
            frame_id=10,
        )
        assert frame["sequence_name"] == "segment_001"
        assert frame["frame_id"] == 10


# ===========================================================================
# 5. Box utilities (pure Python / NumPy, no CUDA)
# ===========================================================================


class TestBoxUtils:
    """Test pure-Python box utility functions."""

    def test_boxes_to_corners_3d_shape(self):
        from detzero_utils.box_utils import boxes_to_corners_3d

        boxes = np.array([[0.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0]], dtype=np.float32)
        corners = boxes_to_corners_3d(boxes)
        # Expected: (N, 8, 3)
        assert corners.shape == (1, 8, 3), f"Unexpected shape: {corners.shape}"

    def test_boxes_to_corners_3d_batch(self):
        from detzero_utils.box_utils import boxes_to_corners_3d

        boxes = np.random.randn(10, 7).astype(np.float32)
        corners = boxes_to_corners_3d(boxes)
        assert corners.shape == (10, 8, 3)


# ===========================================================================
# 6. Version sanity checks
# ===========================================================================


class TestVersions:
    """Ensure version strings were generated by setup.py."""

    def test_detzero_utils_version(self):
        # detzero_utils/__init__.py is empty; version lives in detzero_utils.version
        from detzero_utils import version as _v

        assert hasattr(_v, "__version__")
        assert _v.__version__.startswith("0.1.0")

    def test_detzero_det_version(self):
        import detzero_det

        assert detzero_det.__version__.startswith("0.1.0")

    def test_detzero_track_version(self):
        import detzero_track

        assert detzero_track.__version__.startswith("0.1.0")

    def test_detzero_refine_version(self):
        import detzero_refine

        assert detzero_refine.__version__.startswith("0.1.0")


# ===========================================================================
# 7. Dependency availability checks (informational, not fatal)
# ===========================================================================


class TestDependencies:
    """
    Check whether optional / unlisted dependencies are available.

    These tests are marked xfail so the suite still passes in a minimal
    CPU-only environment; they document what is and is not installed.
    """

    @pytest.mark.xfail(reason="filterpy not listed in requirements.txt; required by tracking Kalman filter", strict=False)
    def test_filterpy_available(self):
        import filterpy.kalman  # noqa: F401

    @pytest.mark.xfail(reason="spconv requires CUDA; CPU-only environment uses a stub", strict=False)
    def test_spconv_available(self):
        import spconv.pytorch as _spconv_pt

        # The conftest stub sets _is_cpu_stub=True; real spconv does not.
        assert not getattr(_spconv_pt, "_is_cpu_stub", False), (
            "Only the CPU stub is present; real spconv (with CUDA) is not installed"
        )

    @pytest.mark.xfail(reason="waymo-open-dataset requires TensorFlow and is not installed in CI", strict=False)
    def test_waymo_open_dataset_available(self):
        import waymo_open_dataset  # noqa: F401

    @pytest.mark.xfail(reason="numba==0.48.0 pinned in requirements.txt may conflict with newer NumPy/Python", strict=False)
    def test_numba_pinned_version(self):
        import numba

        assert numba.__version__ == "0.48.0", (
            f"requirements.txt pins numba==0.48.0 but found {numba.__version__}"
        )
