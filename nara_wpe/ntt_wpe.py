from pathlib import Path
from cached_property import cached_property

import tempfile
import numpy as np
import soundfile as sf
import click
from pymatbridge import Matlab

from nara_wpe import project_root


class NTTWrapper:
    def __init__(self, path_to_pkg, output_path):
        self.path_to_pkg = Path(path_to_pkg)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)

        if not path_to_pkg.exists():
            raise OSError(
                'NTT WPE package does not exist. It has to be downloaded'
                'from http://www.kecl.ntt.co.jp/icl/signal/wpe/download.html')

    @cached_property
    def process(self):
        mlab = Matlab()
        mlab.start()
        return mlab

    def cfg(self, channels):
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
        cfg = self.cfg(channels)

        with tempfile.NamedTemporaryFile() as cfg_file:
            for line in cfg:
                cfg_file.write(line)

            self.process.set_variable("y", y)
            self.process.set_variable("cfg", str(cfg_file))

            assert np.allclose(self.process.get_variable("y"), y)
            assert self.process.get_variable("cfg") == str(cfg_file)

            self.process.run_code("addpath('" + str(cfg_file) + "');")
            self.process.run_code("addpath('" + str(self.path_to_pkg) + "');")

            print("Dereverbing ...")
            msg = self.process.run_code("y = wpe(y, cfg);")

            assert msg['success'] is True, \
                f'WPE has failed. {msg["content"]["stdout"]}'

        y = self.process.get_variable("y")

        # write output
        if len(files) > 1:
            for i, file in enumerate(files):
                sf.write(
                    str(self.output_path / Path(file).name),
                    y[i],
                    samplerate=sampling_rate
                )
        else:
            sf.write(
                str(self.output_path / Path(files).name),
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
    '--output_path',
    default=str(project_root / 'data' / 'dereverberation'),
    help='Output path.'
)
@click.option(
    '--delay',
    default=3,
    help='Delay'
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
def main(path_to_pkg, files, output_path, delay,
         iterations, taps, psd_context):
    """
    A small command line wrapper around the NTT-WPE matlab file.
    http://www.kecl.ntt.co.jp/icl/signal/wpe/
    """
    wrapper = NTTWrapper(path_to_pkg, output_path)
    wrapper(files, delay, iterations, taps, psd_context)


if __name__ == '__main__':
    main()