OBJECTIVE  
Develop a Machine Learning (ML) system capable of solving a prediction or classification problem using Julia language.  
**Implementation**  
For the implementation of the ML system, you should functions from utils/utils_ML1.jl.  
**Selected problem**  
Dataset - Online News Popularity. This dataset summarizes a heterogeneous set of features about articles published by Mashable in a period of two years. The goal is to predict the number of shares in social networks (popularity). No missing values.     
Number of Attributes: 61 (58 predictive attributes, 2 non-predictive, 1 goal field)  

Attribute Information:  
     0. url:                           URL of the article (non-predictive)  
     1. timedelta:                     Days between the article publication and the dataset acquisition (non-predictive)  
     2. n_tokens_title:                Number of words in the title  
     3. n_tokens_content:              Number of words in the content  
     4. n_unique_tokens:               Rate of unique words in the content  
     5. n_non_stop_words:              Rate of non-stop words in the content  
     6. n_non_stop_unique_tokens:      Rate of unique non-stop words in the content  
     7. num_hrefs:                     Number of links  
     8. num_self_hrefs:                Number of links to other articles published by Mashable  
     9. num_imgs:                      Number of images  
    10. num_videos:                    Number of videos  
    11. average_token_length:          Average length of the words in the content  
    12. num_keywords:                  Number of keywords in the metadata  
    13. data_channel_is_lifestyle:     Is data channel 'Lifestyle'?  
    14. data_channel_is_entertainment: Is data channel 'Entertainment'?  
    15. data_channel_is_bus:           Is data channel 'Business'?  
    16. data_channel_is_socmed:        Is data channel 'Social Media'?  
    17. data_channel_is_tech:          Is data channel 'Tech'?  
    18. data_channel_is_world:         Is data channel 'World'?  
    19. kw_min_min:                    Worst keyword (min. shares)  
    20. kw_max_min:                    Worst keyword (max. shares)  
    21. kw_avg_min:                    Worst keyword (avg. shares)  
    22. kw_min_max:                    Best keyword (min. shares)  
    23. kw_max_max:                    Best keyword (max. shares)  
    24. kw_avg_max:                    Best keyword (avg. shares)  
    25. kw_min_avg:                    Avg. keyword (min. shares)  
    26. kw_max_avg:                    Avg. keyword (max. shares)  
    27. kw_avg_avg:                    Avg. keyword (avg. shares)  
    28. self_reference_min_shares:     Min. shares of referenced articles in Mashable  
    29. self_reference_max_shares:     Max. shares of referenced articles in Mashable  
    30. self_reference_avg_sharess:    Avg. shares of referenced articles in Mashable  
    31. weekday_is_monday:             Was the article published on a Monday?  
    32. weekday_is_tuesday:            Was the article published on a Tuesday?  
    33. weekday_is_wednesday:          Was the article published on a Wednesday?  
    34. weekday_is_thursday:           Was the article published on a Thursday?  
    35. weekday_is_friday:             Was the article published on a Friday?  
    36. weekday_is_saturday:           Was the article published on a Saturday?  
    37. weekday_is_sunday:             Was the article published on a Sunday?  
    38. is_weekend:                    Was the article published on the weekend?  
    39. LDA_00:                        Closeness to LDA topic 0  
    40. LDA_01:                        Closeness to LDA topic 1  
    41. LDA_02:                        Closeness to LDA topic 2  
    42. LDA_03:                        Closeness to LDA topic 3  
    43. LDA_04:                        Closeness to LDA topic 4  
    44. global_subjectivity:           Text subjectivity  
    45. global_sentiment_polarity:     Text sentiment polarity  
    46. global_rate_positive_words:    Rate of positive words in the content  
    47. global_rate_negative_words:    Rate of negative words in the content  
    48. rate_positive_words:           Rate of positive words among non-neutral tokens  
    49. rate_negative_words:           Rate of negative words among non-neutral tokens  
    50. avg_positive_polarity:         Avg. polarity of positive words  
    51. min_positive_polarity:         Min. polarity of positive words  
    52. max_positive_polarity:         Max. polarity of positive words  
    53. avg_negative_polarity:         Avg. polarity of negative  words  
    54. min_negative_polarity:         Min. polarity of negative  words  
    55. max_negative_polarity:         Max. polarity of negative  words  
    56. title_subjectivity:            Title subjectivity  
    57. title_sentiment_polarity:      Title polarity  
    58. abs_title_subjectivity:        Absolute subjectivity level  
    59. abs_title_sentiment_polarity:  Absolute polarity level  
    60. shares:                        Number of shares (target)  
  
We treat the problem as a binary classification task. Following the approach used in the original UCI paper, we define an article as popular if its number of shares is above the median value. Otherwise, it is labeled as not popular.  
```
median_shares = median(df[!, colname])  
df.binary_class = df[!, colname] .> median_shares
```  
The problem must be solved using four different approaches.  

**Submission**  
The deliverables must include: 
- The selected dataset. 
- The source code and a detailed report. 
- Auxiliary code developed in previous practices. (utils/utils_ML1.jl)
Folder Structure: 
/root_folder  
├── Report.pdf        (Contains the written report.)  
├── main.jl           (Main Julia script.)  
├── /datasets         (Folder containing the data.)  
└── /utils            (Folder containing auxiliary code.)  
Report contents:   
The report must clearly explain:   
- The selected problem and objectives.  
- The methodology and approaches followed. 
- A discussion and interpretation of the results. 
  
Main.jl - The file should contain the whole code with all approaches of the project. It has to be 
executable from beginning to end, and it must allow to check the same results presented in the report.   
Datasets - Should contain the data used in the different approaches. Within this folder new 
folders can be created, and this structure is up to the decisions made by the work team.  
Utils - It should contain the code from the tutorials in the shape of one or more Julia (.jl) 
file(s).  

**Specific instructions**
1. Report (50% of the project mark)  
- Introduction (10% of the project mark).   
  The introduction must include:  
    - A clear description of the problem to be solved.  
    - A description and summary of the dataset used.   
    - Justification of the evaluation metric(s). 
    -  Explanation of the code structure and organization.  
    -  A short bibliographic review (minimum 3 scientific publications relevant to the problem). 
      - Use a formal citation style suitable for academic publications.  
      - Websites are not considered scientific sources  
      - The description should connect the different works to each other, rather than being just a list of separate, unrelated paragraphs. 
- Development:  
  The students are tasked with investigating various approaches for processing the dataset.  
  The highest achievable score for each attempted approach will be 25% of the total value of this section.  
  Each approach should include:   
    - Dataset description: Even if previously introduced, describe any variations used in each approach (number of samples, features, classes, etc.), supported by relevant graphs or figures.  
    - Data preprocessing: Justify the normalization method and parameters used (min, max, mean, etc.) or explain why normalization was not applied.  
    - Experimental setup: Specify methodology, cross-validation strategy, variable selection dimensionality reduction, etc.  
    - Model experimentation:
    Each approach must test the four ML techniques covered in class:  
        1. Artificial Neural Networks (ANNs): Test at least 8 architectures (1–2 hidden layers).  
        2. Support Vector Machines (SVMs): Test at least 8 configurations with different kernels and values of C.  
        3. Decision Trees: Test at least 6 different maximum depths.  
        4. k-Nearest Neighbors (kNN): Test at least 6 different values of k.  
    The clarity and organization of the explanations is highly valued, as well as the number of experiments.  
    - Ensemble Method:
    Apply at least one ensemble technique (e.g., majority voting, weighted voting, or stacking) combining at least three of the previous individual models.  
    - Results and discussion:  
    Report all experiments using the selected metrics.   
    Include comparisons between models, supported by plots or confusion matrices or statistical significance test (ANOVA, t-test, Friedman, etc.).   
- Final discussion (10% of the project mark)  
  Summarize and evaluate the overall process, comparing the results across different approaches. Highlight the conclusions supported by the experimental findings. 

2. Code (30% of the project mark)  
  The code must: 
    - Set a random seed to ensure reproducibility. 
    - Load and preprocess the dataset. 
    - Extract relevant features. 
    - Split the data (training/testing) 
    - On the train dataset perform a cross-validation using the provided modelCrossValidation function to choose the corresponding parameters. 
    - Train the final model with the full training dataset and evaluate it on the test set. 
    - Include a confusion matrix for evaluation.  
      i. For ANNs, remember to use a validation split within the training data. 