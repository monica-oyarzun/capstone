import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


public_df = pd.read_csv('./data/public-study/EmoTrak-emotrak-2018-04-19T20_50_09.379Z.csv')

# fix 'embarrased' spelling error
# public_df['specificEmotion'] = public_df['specificEmotion'].replace('embarrased', 'embarrassed')

# fill NLP column NaN's with empty string
# public_df[['trigger', 'spendingDayOther', 'otherEmotion']] = public_df[['trigger', 'spendingDayOther', 'otherEmotion']].fillna('')

# split train/test
train, test = train_test_split(public_df, shuffle=True)

# set train/test with reduced variables
train_subset = train[['feelingEmotion', 'generalEmotion', 'specificEmotion',
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
       'selfCarePet', 'selfCareOther', 'selfCareNone', 'selfCareOtherActivity']]

test_subset = test[['feelingEmotion', 'generalEmotion', 'specificEmotion',
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
       'selfCarePet', 'selfCareOther', 'selfCareNone', 'selfCareOtherActivity']]

# replace values to correct for spelling, yes/no -> 1/0, and string 60+ -> int 90
train_subset = train_subset.replace({'embarrased': 'embarrassed', 'yes': 1, 'no': 0, '60+': 90})
test_subset = test_subset.replace({'embarrased': 'embarrassed', 'yes': 1, 'no': 0, '60+': 90})



