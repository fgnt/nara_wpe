from pathlib import Path

import numpy as np
import soundfile as sf
import click
from pymatbridge import Matlab
from librosa.core.audio import resample

from nara_wpe import project_root


@click.command()
@click.option(
    '--path_to_pkg',
    default='/home/danielha/Documents/whk/wpe_v1.32/',
    type=click.Path(exists=True),
)
@click.option(
    '--channels',
    default=8,
    help='Audio Channels D'
)
@click.option(
    '--sampling_rate',
    default=16000,
    help='Sampling rate of audio'
)
@click.option(
    '--file_template',
    default='AMI_WSJ20-Array1-{}_T10c0201.wav',
    help='Audio example. Full path required.'
         ' Included example: AMI_WSJ20-Array1-{}_T10c0201.wav'
)
def main(path_to_pkg, channels, sampling_rate, file_template):
    """
    A small wrapper around the NTT-WPE matlab file.
    """
    path_to_pkg = Path(path_to_pkg)
    cfg = path_to_pkg / 'settings' / 'local.m'

    if not path_to_pkg.exists():
        raise OSError('NTT WPE package does not exist. It has to be downloaded'
                      'from http://www.kecl.ntt.co.jp/icl/signal/wpe/download.html')

    if not cfg.exists():
        raise OSError('Missing config.')

    # load audio
    if file_template == 'AMI_WSJ20-Array1-{}_T10c0201.wav':
        signal_list = [
            sf.read(str(project_root / 'data' / file_template.format(d + 1)))[0]
            for d in range(channels)
            ]
    else:
        signal = sf.read(file_template)[0].transpose(1, 0)
        signal_list = list(signal)
    signal_list = [resample(x_, 16000, sampling_rate) for x_ in signal_list]
    y = np.stack(signal_list, axis=0).transpose(1, 0)


    # Check number of channels and sampling_rate and set local.m accordingly
    modify_settings = False
    lines = []
    with cfg.open() as infile:
        for line in infile:
            if 'num_mic = ' in line and not 'num_out' in line:
                if not str(channels) in line:
                    line = 'num_mic = ' + str(channels) + ";\n"
                    modify_settings = True
            elif 'fs' in line:
                if not str(sampling_rate) in line:
                    line = 'fs =' + str(sampling_rate) + ";\n"
                    modify_settings = True
            lines.append(line)
    if modify_settings:
        with cfg.open('w') as outfile:
            for line in lines:
                outfile.write(line)
        print('Set local.m (config) accordingly to audio file parameters.')

    mlab = Matlab()
    mlab.start()
    mlab.set_variable("y", y)
    mlab.set_variable("cfg", str(cfg))

    assert np.allclose(mlab.get_variable("y"), y)
    assert mlab.get_variable("cfg") == str(cfg)
    
    mlab.run_code("addpath('" + str(cfg) + "');")
    mlab.run_code("addpath('" + str(path_to_pkg) + "');")

    print("Dereverbing ...")
    msg = mlab.run_code("y = wpe(y, cfg);")

    assert msg['success'] is True, 'WPE has failed.'

    y = mlab.get_variable("y")
    print('Finished.')

    sf.write(
        str(project_root / 'data' / 'wpe_out.wav'),
        y[0], samplerate=sampling_rate
    )
    print('Output in {}'.format(str(project_root / 'data' / 'wpe_out.wav')))


if __name__ == '__main__':
    main()






