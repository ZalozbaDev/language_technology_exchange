# coqui_tts_training

Skripty a dokumentacija za TTS trening

# install

git clone git@github.com:ZalozbaDev/TTS.git

install dependencies from "requirements.txt" and then this repo

note: need python < 3.12

# corpora

mkdir corpora
cd corpora

git clone ...

# multi speaker training

- with the current multi-speaker dataset, 500 epochs is more than enough (~ 200000 steps)

# finetuning

- first try with decreased learning rates:

```code
--coqpit.lr_gen 0.00001
--coqpit.lr_disc 0.00001
```

# tensorboard

* append "--bind_all" to tensorboard cmdline to access tensorboard from different PC

