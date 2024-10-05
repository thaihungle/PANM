# PANM
Source code for paper "Plug, Play, and Generalize: Length Extrapolation with Pointer-Augmented Neural Memory"  
- Paper: https://openreview.net/forum?id=dyQ9vFbF6D
- Blog: https://hungleai.substack.com/p/extending-neural-networks-to-new 
- Code reference: https://github.com/thaihungle/SAM  

# Setup  
python 3.8  
```
pip install -r requirements.txt   
mkdir logs
mkdir saved_models
```

# Alogirthm tasks
run training command examples for Copy
``` 
LSTM baseline: python run_algo_task.py -task_json=./tasks/copy.json -model_name=lstm -mode=train 
PANM: python run_toys.py -task_json=./tasks/copy.json -model_name=panm -mode=train  
```

run testing command examples for Copy (x2 test length)
``` 
LSTM baseline: python run_algo_task.py -task_json=./tasks/copy.json -model_name=lstm -mode=test -genlen=2 
PANM: python run_toys.py -task_json=./tasks/copy.json -model_name=panm -mode=test -genlen=2 
```