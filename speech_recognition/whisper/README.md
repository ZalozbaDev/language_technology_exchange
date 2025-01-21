# Training scripts

## Interesting findings

* dilhtwd/0002_train_eval.py:
    * freeze first 15 layers as argued in https://asmp-eurasipjournals.springeropen.com/articles/10.1186/s13636-024-00349-3
    * usage of an optimizer
* fhg/ft_whisper_prep.py:
    * generation of hugging face datasets (faster traininbg, easier publishing of new resources?)
* fhg/ft_whisper_train.py:
    * set language to "czech" for faster convergence


## Outlook

* investigate finetuning of w2v2-bert and Facebook MMS (inspired by SPECOM 2024 paper)
* use augmentation to increase dataset
* enlarge dataset with pseudo labelled resources

# Models

See this page on existing models and the required tooling for GGML conversion:

https://github.com/ZalozbaDev/mudrowak/tree/main/doc/models

