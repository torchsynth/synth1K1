#!/usr/bin/env python3

"""
Script for generating synth1K1
"""

import os
from pathlib import Path
import sys
import argparse
from subprocess import DEVNULL, check_call

import torch
from tqdm import tqdm
import soundfile as sf

from torchsynth.synth import Voice


def generate_synth1k1(output: Path, concat: bool, nebula: str, batches: int):

    voice = Voice(nebula=nebula)

    if torch.cuda.is_available():
        voice.to("cuda")
        print("Running on CUDA")

    output_dir = os.path.abspath(output)
    if not os.path.isdir(output_dir):
        raise ValueError("outdir must be a directory")

    # If batches is 8 then this is synth1K1
    if batches == 8 and nebula == "default":
        basename = "synth1K1"
    else:
        basename = f"torchsynth-voice-{voice.batch_size * batches}"

    if nebula != "default":
        basename = f"{basename}-{nebula}"

    audio_list = []
    with torch.no_grad():
        for i in tqdm(range(batches)):
            audio, params, is_train = voice(i)

            # Write all the audio to disk
            for k in range(len(audio)):
                audio_list.append(audio[k])

                # Save individual audio files if not concatenating
                if not concat:
                    index = voice.batch_size * i + k
                    filename = os.path.join(output_dir, f"{basename}-{index}")

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
        # Concatenate all audio examples together
        pad = int(voice.sample_rate.cpu().item() * 0.1)
        audio_list = [torch.nn.functional.pad(s, (0, pad)) for s in audio_list]
        audio = torch.cat(audio_list)
        filename = os.path.join(output_dir, f"{basename}.wav")
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
    parser.add_argument(
        "--nebula",
        default="default",
        help="Select nebula for synth1K1. ['default', 'drum']",
    )
    parser.add_argument(
        "--batches",
        default=8,
        type=int,
        help="Number of batches to render. Defaults to 8, which is synth1K1",
    )
    args = parser.parse_args(arguments)

    # Create output dir if it doesn't exist
    output = Path(args.output)
    output.mkdir(exist_ok=True)

    # Generate and save the sounds
    generate_synth1k1(output, args.concat, args.nebula, args.batches)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
