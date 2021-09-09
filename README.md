# synth1K1
synth1K1 is a dataset of 1024 4-second long synthesizer sounds rendered using
[torchsynth](https://github.com/torchsynth/torchsynth), a GPU-enabled modular
synthesizer.

This repo contains all the individual sounds from synth1K1 as mp3s in the 
[audio](https://github.com/torchsynth/synth1K1/tree/main/audio) directory. You can
listen to these individual files [here](https://torchsynth.github.io/synth1K1/), or
[on SoundCloud](https://soundcloud.com/user-357924775/synth1k1).

### Generating synth1K1
You can generate synth1K1 yourself by cloning this repo and using the included script.

Install python requirements:
```
python3 -m pip install torchsynth soundfile
```

mp3 rendering requires [LAME](https://lame.sourceforge.io/).

Clone the repo:
```
git clone https://github.com/torchsynth/synth1K1.git
``` 

Run the script to generate audio as mp3s to a directory called `synth1k1`:
```
python3 generate_synth1K1.py ./synth1K1
```

Instead of mp3s, synth1K1 samples can also be concatenated together to create a long
WAV file:
```
python3 generate_synth1K1.py ./synth1K1 --concat
```
