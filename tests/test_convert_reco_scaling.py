"""convert uses the reco id brkraw passes (0.6.0rc2) and brkraw's per-frame
scaling (brkraw.api.scale_frames), and reports a failed NIfTI build instead of
hiding it (BRK-0046). Fake scans only."""

import logging

import numpy as np

from brkraw_dti import hook


class FakeScan:
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail
        self.affine_info = {1: {"affines": np.eye(4)}, 2: {"affines": np.eye(4)}}  # same affine

    def get_nifti1image(self, reco_id, dataobjs, affines, **kwargs):
        self.calls.append({"reco_id": reco_id, "dataobjs": dataobjs, **kwargs})
        if self.fail:
            raise self.fail
        from nibabel.nifti1 import Nifti1Image

        return Nifti1Image(np.asarray(dataobjs[0], dtype=float), affines[0])


def _no_gradients(monkeypatch):
    monkeypatch.setattr(hook, "get_gradients", lambda scan: (None, None))


def test_uses_the_reco_id_brkraw_passes(monkeypatch):
    _no_gradients(monkeypatch)
    monkeypatch.setattr(hook, "_scale_frames", lambda scan, reco_id, data, **kw: (data, False))
    scan = FakeScan()
    hook.convert(scan, np.zeros((2, 2, 1)), np.eye(4), reco_id=2)
    assert scan.calls[0]["reco_id"] == 2


def test_applies_brkraws_per_frame_scaling(monkeypatch):
    _no_gradients(monkeypatch)
    seen = {}

    def fake_scale(scan, reco_id, data, **kw):
        seen.update(reco_id=reco_id, **kw)
        return tuple(np.asarray(d, dtype=float) * 3.0 for d in data), True

    monkeypatch.setattr(hook, "_scale_frames", fake_scale)
    scan = FakeScan()
    out = hook.convert(scan, np.ones((2, 2, 1)), np.eye(4), reco_id=2, axis="dti", frames=[0])
    assert seen == {"reco_id": 2, "axis": "dti", "frames": [0]}
    assert scan.calls[0]["scaling_applied"] is True
    assert np.allclose(np.asarray(out.dataobj), 3.0)


def test_passes_the_legacy_cycle_selection_to_scaling(monkeypatch):
    # brkraw forwards cycle_index/cycle_count to the hook; scale_frames needs them
    # to cut its per-frame values like the data (wi-0038-choi-3)
    _no_gradients(monkeypatch)
    seen = {}

    def fake_scale(scan, reco_id, data, **kw):
        seen.update(reco_id=reco_id, **kw)
        return data, True

    monkeypatch.setattr(hook, "_scale_frames", fake_scale)
    hook.convert(FakeScan(), np.ones((2, 2, 1)), np.eye(4), reco_id=2, cycle_index=1, cycle_count=2)
    assert seen == {"reco_id": 2, "cycle_index": 1, "cycle_count": 2}


def test_a_failed_nifti_build_is_reported(monkeypatch, caplog):
    _no_gradients(monkeypatch)
    monkeypatch.setattr(hook, "_scale_frames", lambda scan, reco_id, data, **kw: (data, False))
    scan = FakeScan(fail=ValueError("VisuCoreDataSlope has 110 values"))
    with caplog.at_level(logging.WARNING):
        hook.convert(scan, np.zeros((2, 2, 1)), np.eye(4), reco_id=2)
    assert "VisuCoreDataSlope has 110 values" in caplog.text


def test_without_a_reco_id_it_still_guesses_for_older_brkraw(monkeypatch):
    _no_gradients(monkeypatch)
    monkeypatch.setattr(hook, "_scale_frames", None)
    scan = FakeScan()
    hook.convert(scan, np.zeros((2, 2, 1)), np.eye(4))
    assert scan.calls and scan.calls[0]["reco_id"] in (1, 2)
    assert "scaling_applied" not in scan.calls[0]
