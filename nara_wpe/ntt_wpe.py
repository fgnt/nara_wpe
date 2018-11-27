from pathlib import Path
from cached_property import cached_property

import tempfile
import numpy as np
import soundfile as sf
import click
from pymatbridge import Matlab

from nara_wpe import project_root


def ntt_wrapper(
        y,
        taps=10,
        delay=3,
        iterations=3,
        sampling_rate=16000,
        path_to_package=project_root / 'cache' / 'wpe_v1.33',
        stft_size=512,
        stft_shift=128
):
    wpe = NTTWrapper(path_to_package)
    return wpe(
        y=y,
        taps=taps,
        delay=delay,
        iterations=iterations,
        sampling_rate=sampling_rate,
        stft_size=stft_size,
        stft_shift=stft_shift
    )


class NTTWrapper:
    """
    The WPE package has to be downloaded from
    http://www.kecl.ntt.co.jp/icl/signal/wpe/download.html. It is recommended
    to store it in the cache directory of Nara-WPE.
    """
    def __init__(self, path_to_pkg):
        self.path_to_pkg = Path(path_to_pkg)

        if not self.path_to_pkg.exists():
            raise OSError(
                'NTT WPE package does not exist. It has to be downloaded'
                'from http://www.kecl.ntt.co.jp/icl/signal/wpe/download.html'
                'and stored in the cache directory of Nara-WPE, preferably.'
            )

    @cached_property
    def process(self):
        mlab = Matlab()
        mlab.start()
        return mlab

    def cfg(self, channels, sampling_rate, iterations, taps,
            stft_size, stft_shift
            ):
        """
        Check settings and set local.m accordingly

        """
        cfg = self.path_to_pkg / 'settings' / 'local.m'
        lines = []
        with cfg.open() as infile:
            for line in infile:
                if 'num_mic = ' in line and 'num_out' not in line:
                    if not str(channels) in line:
                        line = 'num_mic = ' + str(channels) + ";\n"
                elif 'fs' in line:
                    if not str(sampling_rate) in line:
                        line = 'fs =' + str(sampling_rate) + ";\n"
                elif 'channel_setup' in line and 'ssd_param' not in line:
                    if not str(taps) in line and '%' not in line:
                        line = "channel_setup = [" + str(taps) + "; ..." + "\n"
                elif 'ssd_conf' in line:
                    if not str(iterations) in line:
                        line = "ssd_conf = struct('max_iter',"\
                               + str(iterations) + ", ...\n"
                elif 'analym_param' in line:
                    if not str(stft_size) in line:
                        line = "analy_param = struct('win_size',"\
                                + str(stft_size) + ", ..."
                elif 'shift_size' in line:
                    if not str(stft_shift) in line:
                        line = "                      'shift_size',"\
                                + str(stft_shift) + ", ..."
                elif 'hanning' in line:
                    if not str(stft_size) in line:
                        line = "                     'win'       , hanning("\
                                + str(stft_size) + "));"
                lines.append(line)
        return lines

    def __call__(
            self,
            y,
            taps=10,
            delay=3,
            iterations=3,
            sampling_rate=16000,
            stft_size=512,
            stft_shift=128
    ):
        """

        Args:
            y: observation (channels. samples)
            delay:
            iterations:
            taps:
            stft_opts: dict contains size, shift

        Returns: dereverberated observation (channels, samples)

        """

        y = y.transpose(1, 0)
        channels = y.shape[1]
        cfg_lines = self.cfg(
            channels, sampling_rate, iterations, taps, stft_size, stft_shift
        )

        with tempfile.TemporaryDirectory() as tempdir:
            with (Path(tempdir) / 'local.m').open('w') as cfg_file:
                for line in cfg_lines:
                    cfg_file.write(line)

            self.process.set_variable("y", y)
            self.process.set_variable("cfg", cfg_file.name)

            self.process.run_code("addpath('" + str(cfg_file.name) + "');")
            self.process.run_code("addpath('" + str(self.path_to_pkg) + "');")

            msg = self.process.run_code("y = wpe(y, cfg);")
            assert msg['success'] is True, \
                f'WPE has failed. {msg["content"]["stdout"]}'

        y = self.process.get_variable("y")

        return y.transpose(1, 0)


@click.command()
@click.argument(
    'files', nargs=-1,
    type=click.Path(exists=True),
)
@click.option(
    '--path_to_pkg',
    default=str(project_root / 'cache' / 'wpe_v1.33'),
    help='It is recommended to save the '
         'NTT-WPE package in the cache directory.'
)
@click.option(
    '--output_dir',
    default=str(project_root / 'data' / 'dereverberation_ntt'),
    help='Output path.'
)
@click.option(
    '--iterations',
    default=5,
    help='Iterations of WPE'
)
@click.option(
    '--taps',
    default=10,
    help='Number of filter taps of WPE'
)
def main(path_to_pkg, files, output_dir, taps=10, delay=3, iterations=5):
    """
    A small command line wrapper around the NTT-WPE matlab file.
    http://www.kecl.ntt.co.jp/icl/signal/wpe/
    """

    if len(files) > 1:
        signal_list = [
            sf.read(str(file))[0]
            for file in files
        ]
        y = np.stack(signal_list, axis=0)
        sampling_rate = sf.read(str(files[0]))[1]
    else:
        y, sampling_rate = sf.read(files)

    wrapper = NTTWrapper(path_to_pkg)
    x = wrapper(y, delay, iterations, taps,
                sampling_rate, stft_size=512, stft_shift=128
                )

    if len(files) > 1:
        for i, file in enumerate(files):
            sf.write(
                str(Path(output_dir) / Path(file).name),
                x[i],
                samplerate=sampling_rate
            )
    else:
        sf.write(
            str(Path(output_dir) / Path(files).name),
            x,
            samplerate=sampling_rate
        )


if __name__ == '__main__':
    main()
