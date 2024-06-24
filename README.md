# RIR Model Evaluation (WIP)


### 1. Setup
Create and activate the conda env and install the current repo. 
```
pip install -e .
```

### 2. Models
2.1 To run example models, 
```
python -m models.rt2rir.main
```
The result will be saved under `models/rt2rir/result`.

2.2 To add your own model, 
1. Generate RIRs from the following list `datasets/rir_list.csv`.
2. Create your model folder : `models/<YourModelName>/` 
3. Place the generated RIRs in `models/<YourModelName>/result` with the same filename as the original RIRs. Examples can be seen in `models/rt2rir/result`. 

### 3. Rendering test signals 
You can render test signals for as many models as you want. 
Just list them after `--models` like the following:
```
python -m rir_model_eval.render_test_signals --models rt2rir <YourModelName1> <YourModelName2> --output_filename test1
```
The output yaml file and the wav files (ex. test1.yaml and test1 folder) will be inside `output` folder: `output/test1.yaml` and `output/test1/*.wav` 

Each time, this will render a new set of test signals (randomized speakers), so run in N times if you want N variations of the test. 

### 4. Copy the output files into webMUSHRA framework
1. First download the git repo. This is a modified webMUSHRA to support 3AFC testing framework. 
[https://github.com/kyungyunlee/webMUSHRA](https://github.com/kyungyunlee/webMUSHRA)
2. Place the yaml file that was created from above into `config` folder, where all yaml files are. 
3. Move the `output` folder itself (which was created above, containing all the folders with the wav files) into the downloaded webMUSHRA : `webMUSRA/output/test1/*.wav`

