# Airbus Dataset API

## Explanation

The Dataset class acts as an interface to the dataset itself. It takes 3 arguments in its constructor:

|Argument | Description |
|--- | --- |
|train_path | The path to the directory which houses all of the training images |
|test_path | The path to the directory which houses all of the test images |
|masks_path | The path to the directory which houses the train_ship_segmentations.csv file|

The Dataset class has a single exposed function, which returns a generator of batches of specified size.
The batches are sample from the dataset without replacement. The arguments for the draw method are:

|Argument | Description |
|--- | --- |
|size | The number of examples per batch |
|training | A boolean that specifies whether the examples to be drawn are training examples |
|random_state | An int or numpy.random.RandomState to act as a seed for the random number generator|

## Usage
To use this class instantiate the Dataset appropriately and then call draw in the same manner you would use an iterator
(i.e. a for loop). Example:

```
train_path = 'C:/train'
test_path = 'C:/test'
masks_path = 'C:'
dataset = Dataset(train_path, test_path, masks_path)
size = 100
training = True

for batch in dataset.draw(size, training):
    do_work_with_batch(batch)
```

When you want to do a new epoch simply call draw in the manner above again.

Let me know if you find any errors, I'm sure you'll find something :D