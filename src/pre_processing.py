import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


public_df = pd.read_csv('./data/public-study/EmoTrak-emotrak-2018-04-19T20_50_09.379Z.csv')

# replace values to correct for spelling, yes/no -> 1/0, and string 60+ -> int 90
public_df = public_df.replace({'embarrased': 'embarrassed', 'yes': 1, 'no': 0, '60+': 90})

# set 'emotionDuration' to float to be able to get summary stats
public_df['emotionDuration'] = (public_df['emotionDuration']).astype(float)

# convert timestamp columns
public_df['timestamp'] = pd.to_datetime(public_df['timestamp'], unit='s')
public_df['startTime'] = pd.to_datetime(public_df['timestamp'], unit='s')

# replace workPilingUp categories with numericals
workLevels = {0: 0, 'slightly': 1, 'moderately': 2, 'very': 3, 'completely': 4}
public_df['workPilingUp'] = public_df['workPilingUp'].map(workLevels)

# cols to keep
cols_to_keep = ['responseLagSeconds',
       'dailyAlertNumber', 'tiredness',
       'feelingEmotion', 'generalEmotion', 'specificEmotion',
       'intensity', 'generalTrigger', 'specificTrigger',
       'bodySensationIntensity', 'bodySensationForehead', 'bodySensationEyes',
       'bodySensationJaw', 'bodySensationNeck', 'bodySensationShoulders',
       'bodySensationChest', 'bodySensationArms', 'bodySensationHands',
       'bodySensationStomach', 'bodySensationBowel', 'bodySensationLegs',
       'bodySensationFeet', 'bodySensationLowerBack', 'bodySensationUpperBack',
       'bodySensationOther', 'bodySensationOtherLocation',
       'emotionDuration', 'selfCareEatingWell',
       'selfCareCooking', 'selfCareExercise', 'selfCareSeekingSupport',
       'selfCareTimeOutside', 'selfCareSpiritualPractice',
       'selfCareQualityTimeTogether', 'selfCareQualityTimeAlone',
       'selfCarePet', 'selfCareOther', 'selfCareNone', 'selfCareOtherActivity',
       'workPilingUp']


# columns to dummify
cols_to_dummify = ['generalEmotion', 'specificEmotion', 'generalTrigger', 'specificTrigger']

def create_subset(df, cols_to_keep):
    subset = df[cols_to_keep]
    return subset

def dummify_clustering(df, colName):
    colName_cols = pd.get_dummies(df[colName])
    df[colName_cols.columns] = colName_cols
    df.drop([colName], axis=1, inplace=True)

def dummify_regression(df, colName):
    colName_cols = pd.get_dummies(df[colName]).iloc[:,1:]
    df[colName_cols.columns] = colName_cols
    df.drop([colName], axis=1, inplace=True)

# create subset with columns above
subset = create_subset(public_df, cols_to_keep)

# transform NaN's
subset['responseLagSeconds'] = subset['responseLagSeconds'].fillna(subset['responseLagSeconds'].mode()[0])
subset[['generalTrigger', 'specificTrigger', 'workPilingUp']] = subset[['generalTrigger', 'specificTrigger', 'workPilingUp']].fillna(0)
subset.fillna(subset.mean(), inplace=True)

