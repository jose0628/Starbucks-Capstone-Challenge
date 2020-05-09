# Starbucks-Capstone-Challenge

Starbucks of many companies would like to offer personalised promotions to their customers.
Based on the datasets we will analyse the impact and deploy a small predictor for users
who would like to take in advance from a promotion or not.


## Motivation <a name="motivation"></a>



## Project Requirements

* Jupiter Notebook with Python3 and [Anaconda Distribution](https://www.anaconda.com/products/individual) (recommended)

## Files and Descriptions <a name="data"></a>


    .
    ├── Starbucks_Capstone_notebook.ipynb    # Analysis and prediction prototype design
    │   
    ├── scripts                   
    │   ├── starbucks_predictor   # It allows to perform predictions 
    │   └── clean_data.py         # Optimized script to clean the datasets
    │
    ├── data                   
    │   ├── portfolio.json    # Containing offer ids and meta data about each offer (duration, type, etc.) 
    │   ├── profile.json      # Demographic data for each customer
    │   └── transcript.json   # Records for transactions, offers received, offers viewed, and offers completed
    │ 
    └── README.md


Here is the schema and explanation of each variable in the datasets:

**portfolio.json**
* id (string) - offer id
* offer_type (string) - type of offer ie BOGO, discount, informational
* difficulty (int) - minimum required spend to complete an offer
* reward (int) - reward given for completing an offer
* duration (int) - time for offer to be open, in days
* channels (list of strings)

**profile.json**
* age (int) - age of the customer 
* became_member_on (int) - date when customer created an app account
* gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
* id (str) - customer id
* income (float) - customer's income

**transcript.json**
* event (str) - record description (ie transaction, offer received, offer viewed, etc.)
* person (str) - customer id
* time (int) - time in hours since start of test. The data begins at time t=0
* value - (dict of strings) - either an offer id or transaction amount depending on the record


## Creator <a name="author"></a>
* José Mancera [Linkedin](https://www.linkedin.com/in/jose0628/)

## Acknowledgements <a name="ack"></a>
* Udacity Data Scientist Challenge
* Starbucks - Raw Datasets
