# DTSC-3601-Final-Project (NoParking)
Final project repository for DTSC 3602

Our project provides long term prediction of Charlotte traffic patterns surrounding the UNC Charlotte campus, based on historical data. It is primarily intended to benefit those who frequently drive to and from campus, but at different times (like students), so that they can plan their routes ahead of time.

## Installation/Access
We recommend the following libraries as they will be necessary to accurately predict traffic patterns at certain parts of the day. Python 3.10+
is required in order to run this code. The libraries and the versions used in this project are listed below. The other standard libaries do not have a recommended version as they are built into Python.
An environment setup will look like this:
```bash
$ pip install xgboost
$ pip install numpy
$ pip install -U scikit-learn
```
## Required libraries
# Python 3.10+
import requests # 2.34.2
import pandas as pd # 2.3.3
import urllib3 # 2.8.0
import boto3 # 1.43.98
import io
import gzip
import logging
import json
import os
from datetime import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar
from io import StringIO
from your_module import school_holiday_check, preprocess_train_data

## How to download the code
1) Clone the repository using this link: https://github.com/Beansproutss/DTSC-3601-Final-Project.git

```bash
git clone https://github.com/Beansproutss/DTSC-3601-Final-Project.git
cd DTSC-3601.Final_Project
```
2) After cloning, run pip install -r requirements.txt from the NoParking directory

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

Users will be able to input a time/day, and our model will output a map of what traffic should look like. Roads highlighted in orange will show slight traffic, and red will indicate heavy traffic. Using this output, users will be able to plan efficient paths to their destination.

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








