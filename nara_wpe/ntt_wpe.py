from pathlib import Path
from cached_property import cached_property

import tempfile
import numpy as np
import soundfile as sf
import click
from pymatbridge import Matlab

from nara_wpe import project_root


class NTTWrapper:
    def __init__(self, path_to_pkg, output_dir):
        self.path_to_pkg = Path(path_to_pkg)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        if not self.path_to_pkg.exists():
            raise OSError(
                'NTT WPE package does not exist. It has to be downloaded'
                'from http://www.kecl.ntt.co.jp/icl/signal/wpe/download.html')

    @cached_property
    def process(self):
        mlab = Matlab()
        mlab.start()
        return mlab

    def cfg(self, channels, sampling_rate, iterations, taps):
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
                        line = "ssd_conf = struct('max_iter'," + str(iterations) + ", ...\n"
                lines.append(line)
        return lines

    def __call__(self, files, delay, iterations, taps, psd_context):
        """

        Args:
            files: List or single String of input data
            delay:
            iterations:
            taps:
            psd_context:

        Returns:

        """
        # load audio
        if len(files) > 1:
            signal_list = [
                sf.read(str(file))[0]
                for file in files
            ]
            y = np.stack(signal_list, axis=0).transpose(1, 0)
            sampling_rate = sf.read(str(files[0]))[1]
        else:
            y, sampling_rate = sf.read(files)
        channels = y.shape[1]
        cfg_lines = self.cfg(channels, sampling_rate, iterations, taps)

        with tempfile.TemporaryDirectory() as tempdir:
            with (Path(tempdir) / 'local.m').open('w') as cfg_file:
                for line in cfg_lines:
                    cfg_file.write(line)

            self.process.set_variable("y", y)
            self.process.set_variable("cfg", cfg_file.name)

            assert np.allclose(self.process.get_variable("y"), y)
            assert self.process.get_variable("cfg") == cfg_file.name

            self.process.run_code("addpath('" + str(cfg_file.name) + "');")
            self.process.run_code("addpath('" + str(self.path_to_pkg) + "');")

            print("Dereverbing ...")
            msg = self.process.run_code("y = wpe(y, cfg);")

            assert msg['success'] is True, \
                f'WPE has failed. {msg["content"]["stdout"]}'

        y = self.process.get_variable("y")
        y = y.transpose(1, 0)

        # write output
        if len(files) > 1:
            for i, file in enumerate(files):
                sf.write(
                    str(self.output_dir / Path(file).name),
                    y[i],
                    samplerate=sampling_rate
                )
        else:
            sf.write(
                str(self.output_dir / Path(files).name),
                y,
                samplerate=sampling_rate
            )

@click.command()
@click.argument(
    'files', nargs=-1,
    type=click.Path(exists=True),
)
@click.option(
    '--path_to_pkg',
    default=str(project_root / 'cache'),
    help='It is recommended to save the NTT-WPE package in the cache directory.'
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
@click.option(
    '--psd_context',
    default=0,
    help='Left and right hand context'
)
def main(path_to_pkg, files, output_dir, delay,
         iterations, taps, psd_context):
    """
    A small command line wrapper around the NTT-WPE matlab file.
    http://www.kecl.ntt.co.jp/icl/signal/wpe/
    """
    wrapper = NTTWrapper(path_to_pkg, output_dir)
    wrapper(files, delay, iterations, taps, psd_context)


if __name__ == '__main__':
    main()
