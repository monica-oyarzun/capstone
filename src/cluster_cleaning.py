import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# import data
public_df = pd.read_csv('./data/public-study/EmoTrak-emotrak-2018-04-19T20_50_09.379Z.csv')

# show all columns
pd.set_option('display.max_columns', 999)

# fix 'embarrased' spelling error
public_df['specificEmotion'] = public_df['specificEmotion'].replace('embarrased', 'embarrassed')

# convert timestamp columns
public_df['timestamp'] = pd.to_datetime(public_df['timestamp'], unit='s')
public_df['startTime'] = pd.to_datetime(public_df['timestamp'], unit='s')

# body sensation columns
bodySens_cols = ['bodySensationIntensity', 'bodySensationForehead', 'bodySensationEyes',
       'bodySensationJaw', 'bodySensationNeck', 'bodySensationShoulders',
       'bodySensationChest', 'bodySensationArms', 'bodySensationHands',
       'bodySensationStomach', 'bodySensationBowel', 'bodySensationLegs',
       'bodySensationFeet', 'bodySensationLowerBack', 'bodySensationUpperBack',
       'bodySensationOther', 'bodySensationOtherLocation',]

# self care columns
selfCare_cols = ['selfCareEatingWell',
       'selfCareCooking', 'selfCareExercise', 'selfCareSeekingSupport',
       'selfCareTimeOutside', 'selfCareSpiritualPractice',
       'selfCareQualityTimeTogether', 'selfCareQualityTimeAlone',
       'selfCarePet', 'selfCareOther', 'selfCareNone', 'selfCareOtherActivity']

# columns to dummify
cols_to_dummify = ['generalEmotion', 'specificEmotion', 'generalTrigger', 'specificTrigger']

def create_subset(df, cols_to_keep):
    subset = df[cols_to_keep]
    return subset

def dummify(df, colName):
    colName_cols = pd.get_dummies(df[colName])
    df[colName_cols.columns] = colName_cols
    df.drop([colName], axis=1, inplace=True)