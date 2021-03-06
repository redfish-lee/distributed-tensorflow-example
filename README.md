# Distributed Tensorflow Example 

Using data parallelism with shared model parameters while updating parameters asynchronous. See comment for some changes to make the parameter updates synchronous (not sure if the synchronous part is implemented correctly though).

Trains a simple sigmoid Neural Network on MNIST for 20 epochs on three machines using one parameter server. The goal was not to achieve high accuracy but to get to know tensorflow.

Run it like this: 

First, change the hardcoded host names with your own and run the following commands on the respective machines.

```
pc-01$ python dist.py --job_name="ps" --task_index=0 
pc-02$ python dist.py --job_name="ps" --task_index=1
pc-03$ python dist.py --job_name="worker" --task_index=0 
pc-04$ python dist.py --job_name="worker" --task_index=1 
pc-05$ python dist.py --job_name="worker" --task_index=2
```

## Limit GPUs
### in python
```python
os.environ["CUDA_VISIBLE_DEVICES"]="0"
```
### in commmand line
```bash
CUDA_VISIBLE_DEVICES=1 python script.py
```

More details here: [ischlag.github.io](http://ischlag.github.io/)

Thanks to snowsquizy for updating the script to TensorFlow 1.2!
