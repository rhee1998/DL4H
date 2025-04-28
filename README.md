# **(CS 598) Deep Learning for Healthcare: Final Project**

## **📖 Overview of the Project**

### **1. Introduction**

* This repository is the implementation of the final project for **(CS 598) Deep Learning for Healthcare**, Spring 2025 term.

* In this work, we utilized a publicly available [dataset](https://physionet.org/content/dreamt/2.0.0/) related to sleep disorders, introduced in [this study](https://raw.githubusercontent.com/mlresearch/v248/main/assets/wang24a/wang24a.pdf).

* The citation to the original paper is as follows:
```
@inproceedings{wang2024addressing,
  title={Addressing wearable sleep tracking inequity: a new dataset and novel methods for a population with sleep disorders},
  author={Wang, Will Ke and Yang, Jiamu and Hershkovich, Leeor and Jeong, Hayoung and Chen, Bill and Singh, Karnika and Roghanizad, Ali R and Shandhi, Md Mobashir Hasan and Spector, Andrew R and Dunn, Jessilyn},
  booktitle={Proceedings of the fifth Conference on Health, Inference, and Learning (Proceedings of Machine Learning Research, Vol. 248), Tom Pollard, Edward Choi, Pankhuri Singhal, Michael Hughes, Elena Sizikova, Bobak Mortazavi, Irene Chen, Fei Wang, Tasmie Sarker, Matthew McDermott, and Marzyeh Ghassemi (Eds.). PMLR},
  pages={380--396},
  year={2024}
}
```

### **2. What's Already Done?: A Brief Summary of the Original Paper**

* The original paper proposes a machine learning framework that classifies a person's sleep state, using continuous measurements from a wearable device.

* They leverage GPBoost and LightGBM to develop an ***Epoch Classifier*** which classifies a 30-second ***"Epoch"*** into either a ***"SLEEP"*** or ***"WAKE"*** state.

* Furthermore, they introduce...

  * An ***LSTM-Based Post-Processing*** module to allow the integration of long-range temporal relations
 
  * The use of ***Clinical Information*** such as obesity and sleep apnea status to enhance classification peroformance.
 
* As a result, 6 different models were trained, and their performance is shown below:

| Random Effects   | LSTM post-processing | AUROC               | F1 Score              |
|:----------------:|:--------------------:|:-------------------:|:---------------------:|
| None             | No                   | 0.895 ± 0.007       | 0.777 ± 0.009         |
| Obesity          | No                   | 0.902 ± 0.015       | 0.785 ± 0.020         |
| Apnea Severity   | No                   | 0.898 ± 0.016       | 0.782 ± 0.015         |
| None             | Yes                  | 0.915 ± 0.019       | 0.805 ± 0.025         |
| Obesity          | Yes                  | 0.926 ± 0.011       | 0.822 ± 0.019         |
| Apnea Severity   | Yes                  | **0.926 ± 0.016**   | **0.823 ± 0.019**     |


### **3. What's New?: Our Efforts to Extend the Research**

* For the final project, we take three different approaches to extend the results of the original paper

  (1) First, we integrate ***<u>various clinical features</u>*** other than just obesity and apnea severity. Demographics such as age and sex, as well as comorbidity status (chronic diseases, psychological disorders, gastroesophageal reflux disease, etc.) are further included.

  (2) Second, we perform a series of ***<u>SHAP analyses to quantitatively asssess the Epoch Classifier's interpretability</u>***. This was done in search for evidence for the justification of adding clinical features, which, unfortunately, turned out to be unsuccessful.

  (3) Thirdly, we explored the ***<u>impact of window sizes on model performance</u>*** when developing LSTM-based post-processing modules.

* The summary of the final project is given as follows:

<img src="image/Fig_1.png" width="50%">



## **📊 Sample Dataset**

* This section describes a sample dataset



## **✏️ Code Implementation**

### **1. Specification of Dependencies**

Create a new conda environment by running this command:

```
conda env create --file environment_dlh.yaml
```



## **📐 Results**

* ***Copy table & Figure(s) from final report***


## **👏 Contributions**

My NetID is `wounsuk2`, and as this is a solo team, the conceptualization, model development, and writing of the final report was solely done by Wounsuk Rhee.
