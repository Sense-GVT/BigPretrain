
## generate proposal
You can download the proposal file from 
or run `ssearch_imagenet.py` to generate proposal.

## split json
The size of generated proposal file is up to 4G. If the memory is limited, `split_ssearch.py` can help to spit the big file. In ths way, only corresponding proposals are loaded while sampling one batch data.  

