# DTSC-3601-Final-Project (NoParking)
# Final project repository for DTSC 3601 and 3602

Our project provides long term prediction of Charlotte traffic patterns surrounding the UNC Charlotte campus, based on historical data. It is primarily intended to benefit those who frequently drive to and from campus, but at different times (like students), so that they can plan their routes ahead of time.

## Installation/Access
We recommend the following libraries as they will be necessary to accurately predict traffic patterns at certain parts of the day. Python 3.10+ is required in order to run this code. The libraries and the versions used in this project are listed below. The other standard libaries do not have a recommended version as they are built into Python.

# Required libraries that will later be downloaded

Python 3.10+

xgboost

numpy

-U scikit-learn

requests #2.34.2

pandas as pd # 2.3.3

urllib3 # 2.8.0

boto3 # 1.43.98

io

gzip

logging

json

os

datetime

USFederalHolidayCalendar from pandas.tseries.holiday

StringIO from io

## How to download the code
1) Clone the repository using this link: https://github.com/Beansproutss/DTSC-3601-Final-Project.git

In Powershell run these commands to clone, install libraries, and run the pipeline:
Before cloning go to a file in your file system.
Example: cd C:\Users\YourName\Downloads\DTSC

``` 
git clone https://github.com/Beansproutss/DTSC-3601-Final-Project.git
cd DTSC-3601-Final-Project
```
2) After cloning, run pip install -r requirements.txt from the NoParking directory. This will install the required libraries.
```bash
pip install -r requirements.txt 
python bin/run_pipeline.py
```
This will output model evaluation and a sample prediction. If code reviewers want to test their own predictions for different locations/times/days they will have to open VS Code.
Note: this data will only accurately predict congestion for the months of October as the sample data only contains rows from October.

As for the other files in the repository, code reviewers will not be able to run them as they do not have access to our API key, but it is there for them to review and is explained by comments and docstrings. If there is anything they think we should add, we can apply those changes later.

## Usage
"NoParking" can be used for long-term prediction of traffic surrounding the UNC Charlotte campus for different days/times.
The focus were roads surrounding campus such as John Kirk Drive (Near lot 5A), which is listed as ID: 0 in the TomTom API where we got our historical data.
Continued list of included roads and their ID's:

1: University City BLVD #1 Close to Cicis and Goodwill

2: University Village BLVD #2 Close to 7-Eleven

3: Intersection of University City BLVD and East Mallard Creek Church Road

4: Univeristy City BLVD #3 close to East Deck and Circle K

5: Intersection of University City BLVD and Shopping Center Drive

6: East W T Harris BLVD near Don Pedro Mexican Restaurant

7: East W T Harris BLVD near Ashley HomeStore

8: North Tryon St near The WearHouse store

9: Johnson Alumni Way near Shane's Rib Shack

10: North Tryon Street near Chipotle and Atrium Health Urgent Care

Users will be able to input a Location ID, year, month, day, and our model will output whether a road is heavily congested, moderately congested, or low congestion. 
Timestamp should be in this datetime format (year-month-day) with 24 hour clock time system. Formatted in a string.
Example: 2026-10-26 14:00:00

## Get Started Example
To access sample data go to folder labeled sample_data
Please run and read all of the directions above before testing sample_data!!

Open the repository folder in VS Code to create a new file, where you can run our example! (If you cannot find the folder, in your terminal, after using cd to our repo, run "code ." This should open VS Code to the folder!)


import pandas as pd

## 1. Import custom pipeline functions from src/
from src.traffic_pipeline import (
    school_holiday_check,
    combine_dataframes,
    extract_ncdot_traffic_data,
    preprocess_train_data,
    evaluate_model,
    predict_custom_traffic
)

## 2. Load the datasets using your local file paths 
df_tomtom = pd.read_csv(r"your_file_path_sample_data.csv")
ncdot_df = pd.read_csv(r"your_file_path_sample_ncdot_df")

ncdot_df = extract_ncdot_traffic_data(ncdot_df)
df_with_holidays = school_holiday_check(df_tomtom)
combined_df = combine_dataframes(df_with_holidays, ncdot_df)

## 3. to train and save the model once in the notebook:
model, X_test, y_test = preprocess_train_data(combined_df)

## 4. Test any location ID and timestamp
predicted_speed, traffic_condition = predict_custom_traffic(
    model, 
    location_id=2, 
    timestamp_str="2026-10-19 19:00:00"
)

## Contributing
Interested in contributing? Email one of the creators to get started!
## License
‘NoParking’ was created by Isabella Broughton, Caleb Brown, Nirali Patel, and Addy Tran. It is licensed under the terms of the Apache 2.0 license. To see what Apache 2.0 allows in detail go here: https://github.com/Beansproutss/DTSC-3601-Final-Project/blob/main/LICENSE

## Questions
For any questions or suggestions, please contact one of the following creators.
ibrought@charlotte.edu
cbrow407@charlotte.edu
npate231@charlotte.edu
atran50@charlotte.edu








