The model we are trying to create : STGAT (Spatio-Temporal Graph Attention Network)

The required libraries to be installed are provided in 'requirements.txt'.You may create a virtual environment and install all the dependencies using 'pip install -r requirements.txt'.

As of 01/04/2026 : config.py has been coded. Verification successful.
data.py has been coded. Verification successful.

As of 02/04/2026 : model.py, train.py, evaluate.py, main.py coded. Verification successful.

One full run has been carried out. Number of epochs = 200. Output comaparable to that of the reference repository.

Code has been made to use Apple Silicon's GPU. To run elsewhere, change the required parameters in config.py

To run the model, run 'python main.py' post activation of virtual environment. 

One such run has been made, and their results have been saved in the 'results/' folder. 

As of 05/04/2026 : Model has been fine-tuned, improvement in two aspects!
1. Takes 5 seconds per epoch, which takes ~ 17 minutes for training to run to completion ; has been greatly improved (from 27s/epoch, which took around 80-85 minutes).
2. MAPE has gone from 24% - 13%, while showing slight improvement on other metrics too.

Refer to stgat_test_metrics.csv for latest results.

Changes made to config.py and model.py.
