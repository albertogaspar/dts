## How to write a config file

When you want to try different values for the hyperparamters of your model 
(i.e., when you run the experiment using the --grid_search flag) you should write a
yaml file that has values in a list like in the example below.

```yaml
train: [False, True]
dataset: ['uci']
epochs: [200,400,1000]
batch_size: [512, 1024]
input_sequence_length: [384]
output_sequence_length:  [96]
units: [[30],[30,30],[128,512]]
learning_rate: [0.001]
```

#### Few advice: What to do not
- Do not include formulas in the paramters values because it will be interpreted as a string when you parse the yaml file.
  ```yaml
    input_sequence_length: [24*7]
  ``` 
- Do not write numbers using exponential notation
  ```yaml
    learning rate: [1e-3]
  ``` 