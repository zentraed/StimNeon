import numpy as np
import pathlib
import pytest
import stim
import tempfile


@pytest.mark.parametrize("bit_packed", [False, True])
def test_dem_sampler_sample(bit_packed: bool):
    noisy_dem = stim.DetectorErrorModel("""
        error(0.125) D0
        error(0.25) D1
    """)
    noisy_sampler = noisy_dem.compile_sampler()
    det_data, obs_data, err_data = noisy_sampler.sample(shots=100, return_errors=True, bit_packed=bit_packed)
    replay_det_data, replay_obs_data, _ = noisy_sampler.sample(shots=100, recorded_errors_to_replay=err_data, bit_packed=bit_packed)
    np.testing.assert_array_equal(det_data, replay_det_data)
    np.testing.assert_array_equal(obs_data, replay_obs_data)


def test_dem_sampler_sampler_write():
    dem = stim.DetectorErrorModel('''
       error(0) D0
       error(0) D1
       error(0) D0
       error(1) D1 D2 L0
       error(0) D0
    ''')
    sampler = dem.compile_sampler()
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        sampler.sample_write(
            shots=1,
            det_out_file=d / 'dets.01',
            det_out_format='01',
            obs_out_file=d / 'obs.01',
            obs_out_format='01',
            err_out_file=d / 'err.hits',
            err_out_format='hits',
        )
        with open(d / 'dets.01') as f:
            assert f.read() == "011\n"
        with open(d / 'obs.01') as f:
            assert f.read() == "1\n"
        with open(d / 'err.hits') as f:
            assert f.read() == "3\n"

        sampler = stim.DetectorErrorModel('''
           error(1) D0  # this should be overridden by the replay.
           error(1) D1
           error(1) D0
           error(1) D1 D2 L0
           error(1) D0
        ''').compile_sampler()
        sampler.sample_write(
            shots=1,
            det_out_file=d / 'dets.01',
            det_out_format='01',
            obs_out_file=d / 'obs.01',
            obs_out_format='01',
            err_out_file=d / 'err2.01',
            err_out_format='01',
            replay_err_in_file=d / 'err.hits',
            replay_err_in_format='hits',
        )
        with open(d / 'dets.01') as f:
            assert f.read() == "011\n"
        with open(d / 'obs.01') as f:
            assert f.read() == "1\n"
        with open(d / 'err.hits') as f:
            assert f.read() == "3\n"
        with open(d / 'err2.01') as f:
            assert f.read() == "00010\n"


def test_dem_sampler_actually_fills_obs_array():
    dem = stim.DetectorErrorModel('''
       error(1) L0
    ''')
    sampler = dem.compile_sampler()
    _, obs_data, _ = sampler.sample(shots=10000)
    assert np.all(obs_data)


_BOOL_OUTPUT_SIZES = [0, 1, 7, 8, 9, 63, 64, 65, 127, 128, 129, 255, 256, 257, 300]


@pytest.mark.parametrize("num_errors", _BOOL_OUTPUT_SIZES)
def test_bool_output_matches_bit_packed_output(num_errors: int):
    dem = stim.DetectorErrorModel()
    for k in range(num_errors):
        dem.append("error", 0.5, [stim.target_relative_detector_id(k)] + ([stim.target_logical_observable_id(k % 3)] if k % 2 else []))
    num_dets = dem.num_detectors
    num_obs = dem.num_observables
    for shots in _BOOL_OUTPUT_SIZES:
        dets, obs, errs = dem.compile_sampler(seed=shots).sample(shots, return_errors=True)
        pdets, pobs, perrs = dem.compile_sampler(seed=shots).sample(shots, return_errors=True, bit_packed=True)
        for unpacked, packed, n in [(dets, pdets, num_dets), (obs, pobs, num_obs), (errs, perrs, num_errors)]:
            assert unpacked.dtype == np.bool_
            assert unpacked.shape == (shots, n)
            assert unpacked.view(np.uint8).max(initial=0) <= 1
            np.testing.assert_array_equal(unpacked, np.unpackbits(packed, axis=1, bitorder='little')[:, :n])
