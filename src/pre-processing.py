import pandas as pd
import numpy as np

public_df = pd.read_csv('./data/public-study/EmoTrak-emotrak-2018-04-19T20_50_09.379Z.csv')

# fix 'embarrased' spelling error
public_df['specificEmotion'] = public_df['specificEmotion'].replace('embarrased', 'embarrassed')

# fill NLP column NaN's with empty string
public_df[['trigger', 'spendingDayOther', 'otherEmotion']] = public_df[['trigger', 'spendingDayOther', 'otherEmotion']].fillna('')

