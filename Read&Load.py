import pandas as pd

# Set ipython's max row display
pd.set_option('display.max_row', 1000)
# Set iPython's max column width to 50
pd.set_option('display.max_columns', 50)

#Loading .xlsx type data set to data-frame with meaningful name.
df_train = pd.read_excel('Data_Train.xlsx',sheet_name="Sheet1")
df_test = pd.read_excel('Data_Test.xlsx',sheet_name="Sheet1")

#Loading .csv type data set ,considering the data is in same folder, if not then you can refer here for more details
df_csvData = pd.read_csv('horror-train.csv')
