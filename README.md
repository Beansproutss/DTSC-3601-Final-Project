# DTSC-3601-Final-Project (NoParking)
Final project repository for DTSC 3601 and 3602

Our project provides long term prediction of Charlotte traffic patterns surrounding the UNC Charlotte campus, based on historical data. It is primarily intended to benefit those who frequently drive to and from campus, but at different times (like students), so that they can plan their routes ahead of time.

## Installation/Access
We recommend the following libraries as they will be necessary to accurately predict traffic patterns at certain parts of the day. Python 3.10+ is required in order to run this code. The libraries and the versions used in this project are listed below. The other standard libaries do not have a recommended version as they are built into Python.

## Required libraries that will later be downloaded
# Python 3.10+
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

```bash
git clone https://github.com/Beansproutss/DTSC-3601-Final-Project.git
cd DTSC-3601-Final-Project
```
2) After cloning, run pip install -r requirements.txt from the NoParking directory. This will install the required libraries.

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

Users will be able to input a Location ID, time/day, and our model will output whether a road is heavily trafficked, moderately trafficked, or little to no traffic. 

## Get Started Example
import pandas as pd
from src.traffic_pipeline.py import school_holiday_check, preprocess_train_data
# 1. Load the pre-extracted sample dataset included in the repo bin folder
sample_df = pd.read_csv("sample_data.csv")

sample_ncdot_df = pd.read_csv("sample_ncdot_data.csv")

# 2. Run your functions directly on it
df_with_holidays = school_holiday_check(df)
model = preprocess_train_data(df_with_holidays)

print(evaluate_mode(model))
print("Model sucessfully trained on sample data!)


## Contributing
Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License
‘NoParking’ was created by Isabella Broughton, Caleb Brown, Nirali Patel, and Addy Tran. It is licensed under the terms of the Apache 2.0 license. To see what Apache 2.0 allows in detail go here: https://github.com/Beansproutss/DTSC-3601-Final-Project/blob/main/LICENSE

## Questions
For any questions or suggestions, please contact one of the following creators.
ibrought@charlotte.edu
cbrow407@charlotte.edu
npate231@charlotte.edu
atran50@charlotte.edu








