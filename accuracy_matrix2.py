import matplotlib.pyplot as plt 
from sklearn.datasets import make_circles #importamos dataset
from sklearn.model_selection import train_test_split #importamos libreria para separar las x y las y 
from sklearn.preprocessing import StandardScaler #importamos para estandarizar los numeros asi trabaja mejor la red 
from sklearn.neural_network import MLPClassifier #Red neuronal 
from sklearn.metrics import (confusion_matrix, accuracy_score, classification_report)

def main():
    x, y = make_circles()
    plt.scatter(x[:,0], x[:,1], c=y)
    plt.grid()
    plt.show()

    x_train, x_test, y_train, y_test = train_test_split(x,y) #Dividimos los datos 

    #Estandarizamos los datos 
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    #Creamos el modelo y lo entrenamos
    clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    clf.fit(x_train,y_train)

    #Evaluamos el modelo 
    y_pred = clf.predict(x_test)

    #mostramos los resultados
    print(f'Accuracy: {accuracy_score(y_test,y_pred)}')
    print('Confusion matrix:\n', confusion_matrix(y_test,y_pred))
    print('clasification report:\n', classification_report(y_test,y_pred))





if __name__ == '__main__':
    main()