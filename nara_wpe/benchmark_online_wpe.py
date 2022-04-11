# ToDo: move this file to tests

import sys
from itertools import product

import pandas as pd
if sys.version_info < (3, 7):
    import tensorflow as tf

from nara_wpe import tf_wpe

if sys.version_info < (3, 7):
    benchmark = tf.test.Benchmark()
configs = []
delay = 1


def config_iterator():
    return product(
        range(1, 11),
        [5, 10],  # K
        [2, 4, 6],  # num_mics # range(2, 11)
        [512],  # frame_size # 1024
        [tf.complex64],  # dtype # , tf.complex128
        ['/cpu:0']  # device # '/gpu:0'
    )


if __name__ == '__main__' and sys.version_info < (3, 7):
    print('Generating configs...')
    for repetition, K, num_mics, frame_size, dtype, device in config_iterator():
        inv_cov_tm1 = tf.eye(
            num_mics * K, batch_shape=[frame_size // 2 + 1], dtype=tf.complex64)
        filter_taps_tm1 = tf.zeros(
            (frame_size // 2 + 1, num_mics * K, num_mics), dtype=tf.complex64)
        input_buffer = tf.zeros(
            (K + delay + 1, frame_size // 2 + 1, num_mics), dtype=tf.complex64)
        power_estimate = tf.ones((frame_size // 2 + 1,), dtype=tf.complex64)
        with tf.device(device):
            configs.append(dict(
                repetition=repetition,
                K=K,
                num_mics=num_mics,
                frame_size=frame_size,
                dtype=dtype,
                device=device,
                op=tf_wpe.online_wpe_step(
                    input_buffer, power_estimate, inv_cov_tm1, filter_taps_tm1,
                    0.9999, K, delay)
            ))

    print('Benchmarking...')
    results = []
    with tf.Session() as sess:
        for cfg in configs:
            print(cfg)
            result = benchmark.run_op_benchmark(
                sess,
                cfg['op'],
                min_iters=100
            )
            result['repetition'] = cfg['repetition']
            result['K'] = cfg['K']
            result['num_mics'] = cfg['num_mics']
            result['frame_size'] = cfg['frame_size']
            result['device'] = cfg['device']
            result['dtype'] = cfg['dtype'].name
            result['fps'] = 1 / result['wall_time']
            result['real_time_factor'] = (
                (16000 / result['frame_size']) * 4 / result['fps']
            )
            results.append(result)

    res = pd.DataFrame(results)
    print(res.groupby(['K', 'num_mics', 'frame_size', 'device', 'dtype']).mean())

    with open('online_wpe_results.json', 'w') as fid:
        res.to_json(fid)
