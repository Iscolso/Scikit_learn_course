#Dividimos datos de entrenamiento y de para evaluar

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def main():
    iris = load_iris()

    x_train, x_test, y_train, t_test = train_test_split(iris.data, iris.target) 
    #Normalmente hace un 75% de division y los revuelve, esto se puede variar con test_size= 

    #shuffle=False si no queremos que se haga un barajeo a la division, pero no es recomendable 

    #En caso de tener por columnas simplemente seleccionamos las columnas que queremos y las dividimos en x y y 


if __name__ == '__main__':
    main()