import numpy as np 
import pandas as pd 
#import keras 
import matplotlib.pyplot as plt 
import seaborn as sns 

def main():
    taxi_df = pd.read_csv('https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv')

    training_df = taxi_df[['TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE']]

    training_df.corr(numeric_only=True)

    sns.pairplot(training_df, x_vars=["FARE", "TRIP_MILES", "TRIP_SECONDS"], y_vars=["FARE", "TRIP_MILES", "TRIP_SECONDS"])
    plt.show()

if __name__ == '__main__':
    main()