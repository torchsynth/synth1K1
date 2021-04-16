#!/usr/bin/env python3

"""
Script for generating synth1K1
"""

import os
import sys
import argparse

import numpy as np
import torch
from tqdm import tqdm
from pydub import AudioSegment

from torchsynth.synth import Voice


def generate_synth1k1(output):

    voice = Voice()

    output_dir = os.path.abspath(output)
    if not os.path.isdir(output_dir):
        raise ValueError("outdir must be a directory")

    with torch.no_grad():
        for i in tqdm(range(8)):
            audio = voice(i)

            # Write all the audio to disk
            for k in range(len(audio)):
                index = voice.batch_size * i + k
                filename = os.path.join(output_dir, f"synth1K1-{index}")
                audio_data = audio[k].numpy().astype(np.float32)
                audio_data *= 2147483647
                audio_data = audio_data.astype(np.int32)
                sound = AudioSegment(
                    audio_data.tobytes(),
                    frame_rate=int(voice.sample_rate.item()),
                    sample_width=4,
                    channels=1
                )
                sound.export(f"{filename}.mp3", format="mp3")


def main(arguments):

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('outdir', help="Output directory for audio files", type=str)
    args = parser.parse_args(arguments)

    generate_synth1k1(args.outdir)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
