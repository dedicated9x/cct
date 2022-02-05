Step 1. Environment installation.
```
conda env create -f env.yaml
```

Step 2. Dataset downloading.
It should be done manually. Expected result is as follows 
```
data/
|-- datasplits.mat
|-- 17flowers
|   |-- jpg
|   |   |-- files.txt
|   |   |-- image_0001.jpg
|   |   |-- image_0002.jpg
|   |   |-- image_0003.jpg
...
```

Step 3. Training
```
python run_experiment.py
```