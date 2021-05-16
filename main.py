import pandas as pd


df = pd.read_json('reviews_Movies_and_TV_5.json', lines=True)
#df.info()

dfID = df.groupby(df.reviewerID)
print(dfID.get_group("AQP1VPK16SVWM"))

training_data = dfID.sample(frac=0.8, random_state=1)
print(training_data.groupby(training_data.reviewerID).get_group("AQP1VPK16SVWM"))

testing_data = df.drop(training_data.index)
print(testing_data.groupby(testing_data.reviewerID).get_group("AQP1VPK16SVWM"))

# print(f"No. of training examples: {training_data.shape[0]}")
# print(f"No. of testing examples: {testing_data.shape[0]}")
