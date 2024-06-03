import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

def main():
    # Loading the airline data
    dataFilePath = 'airline.arff'
    airlineData, meta = arff.loadarff(dataFilePath)

    # Initialising the data frame
    dataFrame = pd.DataFrame(airlineData)
    dataFrame['Date'] = pd.to_datetime(dataFrame['Date'])

    # Seperating the date column and passenger numbers column 
    dateColumn = dataFrame['Date']
    passengerNumbers = dataFrame.drop(['Date'], axis=1)

    # Applying Min Max to passenger numbers
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaledData = scaler.fit_transform(passengerNumbers)

    # Creating a new data frame with scaled passenger numbers and date column
    dataFrameScaled = pd.DataFrame(scaledData, columns=passengerNumbers.columns)
    dataFrameScaled['Date'] = dateColumn

    # Using the data frame with Linear Regression Model for 3 different periods
    linearRegressionModel(dataFrameScaled,10)
    linearRegressionModel(dataFrameScaled,20)
    linearRegressionModel(dataFrameScaled,30)

def linearRegressionModel(dataFrame,period):
  # Creating dataset with period
  x,y = createDatasetWithPeriod(dataFrame, period)
  
  # Initializing the train and test data
  xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)

  # Initializing the Linear Regression Model
  model = LinearRegression()

  # Training the model
  model.fit(xTrain, yTrain)

  # Calculating the mean squared error
  yPred = model.predict(xTest)
  mse = mean_squared_error(yTest, yPred)

  print(f"Linear Regression Mean Squared Error for period of {period}: {mse}")

def createDatasetWithPeriod(dataFrame, period):
  # Seperating passenger numbers from the data frame
  dataFramePassNum = dataFrame['passenger_numbers']

  # Converting the passenger numbers frame to numpy array
  numpyFromDataFrame = dataFramePassNum.to_numpy()

  x = []
  y = []

  for i in range(len(numpyFromDataFrame)-period):
    # Getting the next n values for X as a row, and n+1th value for y as a target value
    x.append(numpyFromDataFrame[i:i+period])
    targetValue = numpyFromDataFrame[i+period]
    y.append(targetValue)
    # Shifting by 1 until the end of the array

  return np.array(x), np.array(y)

main()