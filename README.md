# State of the Art
pytorch reference implementations of adaptive resonance theory. one file per model, no framework.
 
- `art1.py` — ART 1 (Carpenter & Grossberg, 1987), binary inputs
```python
layer = ART1(196, 1024, vigilance=0.4)
 
layer.train(); layer(train_x) 
layer.eval();  layer(test_x).category 
```
 
`vigilance` low gives a few categories, high gives one per sample.
 
`python art1.py` runs it on 14x14 binarized MNIST. 


<img width="2739" height="1475" alt="art" src="https://github.com/user-attachments/assets/f202b602-1a15-4212-b0af-b34b7ea8dffb" />

more models later.
