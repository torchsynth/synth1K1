#!/usr/bin/env python3

"""
Script for generating synth1K1
"""

import os
import sys
import argparse
from subprocess import DEVNULL, check_call

import torch
from tqdm import tqdm
import soundfile as sf

from torchsynth.synth import Voice
import torchsynth.util as util


def generate_synth1k1(output, concat):

    voice = Voice()

    if torch.cuda.is_available():
        voice.to("cuda")
        print("Running on CUDA")

    output_dir = os.path.abspath(output)
    if not os.path.isdir(output_dir):
        raise ValueError("outdir must be a directory")

    audio_list = []
    with torch.no_grad():
        for i in tqdm(range(8)):
            audio = voice(i)

            # Write all the audio to disk
            for k in range(len(audio)):
                audio_list.append(audio[k])

                # Save individual audio files if not concatenating
                if not concat:
                    index = voice.batch_size * i + k
                    filename = os.path.join(output_dir, f"synth1K1-{index}")

                    # Write WAV file
                    wav_file = f"{filename}.wav"
                    sf.write(
                        wav_file,
                        audio[k].cpu().numpy(),
                        voice.sample_rate.cpu().int().item(),
                    )

                    # Convert to mp3 then delete WAV
                    check_call(
                        ["lame", "-V0", f"{wav_file}", f"{filename}.mp3"],
                        stdout=DEVNULL,
                        stderr=DEVNULL,
                    )
                    os.remove(wav_file)

    if concat:
        pad = int(voice.sample_rate.cpu().item() * 0.1)
        audio_list = [
            torch.nn.functional.pad(s, (0, pad)) for s in audio_list
        ]
        audio = torch.cat(audio_list)
        filename = os.path.join(output_dir, "synth1K1.wav")
        sf.write(filename, audio.cpu().numpy(), voice.sample_rate.cpu().int())


def main(arguments):

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("output", help="Output directory", type=str)
    parser.add_argument(
        "--concat",
        default=False,
        action="store_true",
        help="Store as single file concatenated together",
    )
    args = parser.parse_args(arguments)

    generate_synth1k1(args.output, args.concat)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
