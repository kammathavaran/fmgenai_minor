The repo contains code and data for answering all the questions in the exam. 


Structure of the repo 

base
    code_files.py... 
    DATA.md
    requirements.txt 
    LLM_Usage.md
    prompts
        - prompts and data files 
    results 
        - results of the tests 
    images
        - images used in the report 


Instructions 
Note that for most of the tests in this repo, you need to run an independent llm model (as mentioned inthe tests) locally. The openAI based REST end point can be configured for each file as LMSTUDIO_API_URL. 


## Part A - Question 3 

# calculate_basicmetrics.py 
This code is used to generate BLEU, METEOR and ROUGE from the 12k long context file derived from the LongAlapca-12k dataset mentioned in the paper. The code will create the model_comparison_with_metrics.csv file in results 

Usage: python3 calculate_basicmetrics.py 


## Part A - Question 4 

# longlora_pubmed.py
This code is used to generate Accuracy, Macro-F1, BLEU, METEOR and ROUGE from the custom PubMed data file in prompts/custom_pubmed_dataset.json Generates the results as evaluation_metric_*.csv and evaluation_metrics_*.json file in results folder. 

Usage: python3 longlora_pubmed.py

## Part B - Question 1 

# code_switch_test.py
This code is used to simulate Code‐Switch stress test by using Hindi, English and Hinglish along with code switch test. The data file for this test is prompts/multi-language-questions-set.csv. This file generates raw_predictions.csv report under the results folder. 

Usage: python3 code_switch_test.py



## Part B - Question 3 

# find_passphrase.py 
This code is used to find the passphrase from a set of long context data as described in the passphrase question. Will generate the sliding_window_results.csv and baseline_results.csv along with corresponding json files. 

Usage: python3 find_passphrase.py
