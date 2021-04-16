#!/usr/bin/env python3

"""
Script for generating synth1K1
"""

import os
import sys
import argparse

import torch
from tqdm import tqdm
import soundfile as sf

from torchsynth.config import SynthConfig
from torchsynth.synth import Voice


def generate_synth1K1(output):

    voice = Voice()

    if torch.cuda.is_available():
        voice.to('cuda')

    output_dir = os.path.abspath(output)
    if not os.path.isdir(output_dir):
        raise ValueError("outdir must be a directory")

    with torch.no_grad():
        for i in tqdm(range(8)):
            audio = voice(i)

            # Write all the audio to disk
            for k in range(len(audio)):
                index = voice.batch_size * i + k
                filename = os.path.join(output_dir, f"synth1K1-{index}.ogg")
                sf.write(filename, audio[k], int(voice.sample_rate.item()))


def main(arguments):

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('outdir', help="Output directory for audio files", type=str)
    args = parser.parse_args(arguments)

    generate_synth1K1(args.outdir)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
