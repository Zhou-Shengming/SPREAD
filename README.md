# SPREAD
## About SPREAD

SPREAD is a framework forrecognizing promoters in Pseudomonas aeruginosa.

The datasets can be found in `./data/`. The SPREAD models is available in `./model/`. The prediction code can be found in `SPREAD.py`. 

We use 'N' to represent the absence in the sequences.

### Requirements:

- python 3.7
- Keras==2.1.0
- numpy==1.18.0
- scikit-learn==0.20.1
- tensorflow==2.3.0

# Usage
```
python SPREAD.py --input example.fasta --output result.txt
```

- '--input' query sequences to be predicted in fasta format.
- '--output' save the prediction results.
