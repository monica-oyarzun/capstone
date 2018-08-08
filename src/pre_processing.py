import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


public_df = pd.read_csv('./data/public-study/EmoTrak-emotrak-2018-04-19T20_50_09.379Z.csv')

# replace values to correct for spelling, yes/no -> 1/0, and string 60+ -> int 90
public_df = public_df.replace({'embarrased': 'embarrassed', 'yes': 1, 'no': 0, '60+': 90})

# convert timestamp columns
public_df['timestamp'] = pd.to_datetime(public_df['timestamp'], unit='s')
public_df['startTime'] = pd.to_datetime(public_df['timestamp'], unit='s')

# replace workPilingUp categories with numericals
workLevels = {0: 0, 'slightly': 1, 'moderately': 2, 'very': 3, 'completely': 4}
public_df['workPilingUp'] = public_df['workPilingUp'].map(workLevels)

# fill NLP column NaN's with empty string
# public_df[['trigger', 'spendingDayOther', 'otherEmotion']] = public_df[['trigger', 'spendingDayOther', 'otherEmotion']].fillna('')

# split train/test
train, test = train_test_split(public_df, shuffle=True)

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

def dummify(df, colName):
    colName_cols = pd.get_dummies(df[colName])
    df[colName_cols.columns] = colName_cols
    df.drop([colName], axis=1, inplace=True)


