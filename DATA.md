The new data for evaluating the model used is from Pubmed. 

https://github.com/pubmedqa/pubmedqa/tree/master?tab=readme-ov-file


My approach is to select a dataset from a different domain. I decided to use PubMedQA dataset, which is a biomedical QA dataset with easy analysis (yes/no) and results set for detailed analysis of the response. 

This BioMed paper is a great choice to test the model because the paper dealt with testing this in various different domains other than biomedical literature. 


# Dataset provenance & license
Data is licensed under MIT license and available on github. This collection consists of three data categories. 
PQA-L  - Labelled by humans and can be used for evaluation. 
PQA-U - Unlabeled and can be used for un-supervised learning. 
PQA-A - Heuristically created questions used for pre-training. 
We will select PQA-L. This dataset is of 1000 entities and has a question, multiple contexts to derive an answer from, a long answer and a short yes/no answer for conclusion. This dataset is fully annotated and human verified. 

# Hypothesized shift
Given the dataset is of longer context than other QA datasets, I feel LongLoRA will perform better than other models. However due to the nature of the literature, which has dense information in a more terse language than the data LongLoRA was tested against earlier, there might be shifts in the results. This dataset needs better reasoning based on abstract constructs for a model to fare better. My prediction is that the model will show degraded results than what was published before particularly for the yes/no answers (because it needs reasoning). 

# Custom dataset: 

A subset of 60 items of PQA-L (first 60 was selected) will be used for the evaluation. 
Metrics: To follow the paper and compare results, we need to calculate the perplexity score at higher context lengths. However due to computational limitations, I decided to do evaluations on an already fine tuned model created by the authors. The maximum context length will be 13k. However the dataset has a smaller context length. The metrics that will be derived will be basic metrics of BLEU, METEOR and ROUGE as taught in the class. Accuracy and Macro-F1 will also be measured across for short yes/no answers in the data. 

# Data types                              Metrics 
Long form answer                          BLEU, METEOR and ROUGE
Conclusive short yes/no answers           Accuracy and Macro-F1


# Prompt design:

The dataset has two different sets of prompts to set contextual understanding for reasoning. Both will be concatenated and used. Additional prompting for deriving yes/no answers will be done parallely with the contextual long answer test. 
