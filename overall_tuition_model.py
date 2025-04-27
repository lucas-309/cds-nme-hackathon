import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import joblib

def loadData():
   return pd.read_csv("archive/overall_tuition.csv")

def pivotData(df):
    pivot = (
        df.pivot_table(
            index=["Year", "State", "Type", "Length"],   # keys that define one “entry”
            columns="Expense",
            values="Value",
            aggfunc="first",
            fill_value = 0
        )
        .reset_index()
    )

    pivot["Total_cost"] = pivot["Fees/Tuition"] + pivot["Room/Board"]
    pivot["Year_scaled"] = pivot["Year"] - 2000

    pivot = pivot.drop(['Year', 'Fees/Tuition', 'Room/Board'], axis=1).copy()

    pivot['Length'] = pivot['Length'].str[0]
    pivot['Length'] = pivot['Length'].astype(int)
    return pivot

def removeOutliers(df):
    Q1 = df.Total_cost.quantile(0.25)
    Q3 = df.Total_cost.quantile(0.75)
    IQR = Q3 - Q1
    df = df.drop(df.loc[df['Total_cost'] > (Q3 + 1.5 * IQR)].index)
    df = df.drop(df.loc[df['Total_cost'] < (Q1 - 1.5 * IQR)].index)
    return df

# Drop outlier by IQR calculation
def encodeData(df):
    df = pd.get_dummies(df, prefix='Type', columns=['Type'], drop_first=False)
    df = pd.get_dummies(df, prefix='State', columns=['State'], drop_first=False)
    return df

def createInput(year, state, type, length):
    df = loadData()
    df = pivotData(df)
    df = encodeData(df)
    x = df.iloc[:, :1].join(df.iloc[:, 2:])
    template_data = {col: 0 for col in x.columns}
    template_data["Year_scaled"] = year - 2000
    template_data["State_" + state] = 1
    template_data["Type_" + type] = 1
    template_data["Length"] = int(length[0])
    return pd.DataFrame([template_data])

df = loadData()
pivot = pivotData(df)
pivot = removeOutliers(pivot)
pivot = encodeData(pivot)

x = pivot.iloc[:, :1].join(pivot.iloc[:, 2:])
y = pivot.iloc[:,1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

reg = LinearRegression()
reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)

r2 = r2_score(y_test, y_pred)

# Testing the model with future years
# test_data = {col: 0 for col in x_train.columns}
# test_data["Year_scaled"] = 25
# test_data["Type_Private"] = 1
# test_data["Type_Public In-State"] = 0
# test_data["Type_Public Out-of-State"] = 0
# test_data["Length"] = 4
# test_df = pd.DataFrame([test_data])
# print(reg.predict(test_df))

# print(reg.predict(createInput(2030, "California", "Private", "4-year")))

joblib.dump(reg, "reg_model.pkl")

# Note: the dataset uses average costs, so predictions are on the lower end