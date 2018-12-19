## Group: THREE COMMA CLUB (Efe Acer, Murat Topak, Daniil Dmitriev)

### Steps to reproduce our result on the CrowdAI platform:
	
1) Make sure the raw datasets `data_train.csv` and `sample_submission.csv` are inside the 
	`datasets` folder located here in the `scripts` folder.
	
2) Make sure that  you have the required libraries:
* Default libraries
  + os
  + collections
  + cython
* Libraries installed via `pip`
  + NumPy
  + SciPy
  + Pandas
  + scikit-learn
* Open source libraries with specific installation
  + Surprise Library
    + Can be founde [in this GitHub repository](https://github.com/NicolasHug/Surprise). 
    + Please install it using the installation directives in the repository
    + or using the commands:
    ```
    git clone https://github.com/NicolasHug/surprise.git
    python setup.py install
    ```

3) Run the project following the steps:
  * Open a command line and navigate to this `scripts` folder.
  * * Execute `run.py` using the command:
       ```
       python -m run
       ```
     * or run the notebook `run.ipynb`
  * Note: run.py and run.ipynb uses hard coded blending weights, the weights can be computed
    by following the previous instruction for `run_blending.py` or `run_blending.ipynb`.
    
4) The predictions of the final model will be created in a file named `final_submission.csv`,
located inside the `datasets` folder. (The procedure takes around half an hour in total.)

(`python --version` -> `Python 3.6.5 :: Anaconda, Inc.`)

### File organization

- *datasets* :  contains the trainining and test sets.
    - *data_train.csv* : training set
    - *sample_submission.csv* : test set provided by CrowdAI
    - *final_submission.csv* : our final submission to CrowdAI
- *data_related*: contains useful codes to process and analyze raw data
    - *data.py*
    - *data_processing.py*
- *models* : contains all individual models used in the project
    - *baselines.py*
    - *MF.py* : abstract class for Matrix Factorization based models
    - *MF_ALS.py*
    - *MF_SGD.py*
    - *MF_BSGD.py*
    - *MF_implicit.py*
    - *surprise_models.py* : Surprise Library wrapper
- *helpers* : contains helper functions 
    - *loss_functions.py*
- *run.py*
- *run.ipynb* 
- *run_blending.py* 
- *run_blending.ipynb*
- *blending.py*
- *data_exploration.ipynb* : Notebook that provides the figures used in the report