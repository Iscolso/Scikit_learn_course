import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LinearRegression

def main():
    df = pd.read_csv('data3.csv')
    pd.options.display.max_rows = 9999
    df.dropna(inplace=True)

    #sns.scatterplot(x='Duration', y='Calories', data=df, hue='Calories', palette='YlGnBu')
    #plt.show()

    x = df[['Duration']]
    y=df[['Calories']]
    clf = LinearRegression()
    clf.fit(x,y) #Con este comando entrenamos el modelo de regresion lineal
    print(clf.coef_) #mostramos los coeficientes del modelo 
    print(clf.intercept_)


    #Graficamos 
    plt.scatter(x,y)
    plt.plot(x,clf.predict(x))
    plt.title('Regresion lineal simple')
    plt.xlabel('Duracion en horas')
    plt.ylabel('Calorias quemadas')
    plt.legend(['y', 'predicciones'])
    plt.grid()
    plt.show()
if __name__ == '__main__':
    main()