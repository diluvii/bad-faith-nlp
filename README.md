## A brief explanation of where all the code is

```
bad-faith-nlp/
└── src/
    ├── main_rdg.py             # main entrypoint for retrieve-delete-generate (li et al., 2017)
    ├── corpora.py              # helper for constructing corpora (yelp, stackexchange, r/aita, sarcasm)
    ├── corpora_new.py          # helper for constructing second corpora (r/aita)
    ├── main_llm.py             # main entrypoint for seq2seq implementation
    ├── prepare_llm_data.py     # helpers for data processing for the above
    └── train.py                # training the above seq2seq model
```

To run it, create a virtual environment and install packages as specified by `requirements.txt`, `cd` into `src`, and run either `python3 main_rdg.py` or `python3 main_llm.py` as entry points to the implementations. You can change the code to edit configurations
