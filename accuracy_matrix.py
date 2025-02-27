import matplotlib.pyplot as plt 
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier #Una red neuronal convencional 
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report)

def main():
    x, y = make_moons(n_samples=200)

    plt.scatter(x[:,0], x[:,1], c=y)
    plt.grid()
    plt.show()

    #Dividimos nuestros datos con el train test 
    x_train, x_test, y_train, y_test = train_test_split(x,y)
    
    #Estandarizamos asi el modelo opera con mejor precision 
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)


    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500) #Capa oculta de 100 neuronas 
    mlp.fit(x_train, y_train)
    #Evaluamos el modelo 
    y_pred = mlp.predict(x_test)
    
     
    print(f'Acuraccy: {accuracy_score(y_test,y_pred):.2f}')
    print('confussion matrix:\n', confusion_matrix(y_test,y_pred))
    print('Clasification report:\n', classification_report(y_test,y_pred)  )
if __name__ == '__main__':
    main()