from sklearn.datasets import load_wine #importamos dataset
from sklearn.model_selection import train_test_split #importamos libreria para separar las x y las y 
from sklearn.preprocessing import StandardScaler #importamos para estandarizar los numeros asi trabaja mejor la red 
from sklearn.neural_network import MLPClassifier #Red neuronal 
from sklearn.metrics import (confusion_matrix, accuracy_score, classification_report)

def main():
    wine = load_wine()

    x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target)

    standar = StandardScaler()
    x_train = standar.fit_transform(x_train)
    x_test = standar.fit_transform(x_test)

    clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    print(f'Acuraccy: {accuracy_score(y_test,y_pred):.2f}')
    print('confussion matrix:\n', confusion_matrix(y_test,y_pred))
    print('Clasification report:\n', classification_report(y_test,y_pred)  )


if __name__ == '__main__':
    main()