"""
TensorFlow - Klasyfikacja i Regresja

Opis problemu:
  - Aplikacja demonstruje wszechstronne zastosowanie biblioteki TensorFlow/Keras w uczeniu maszynowym.
  - Zadanie 1 (Banknoty): Porównuje klasyczne algorytmy (Drzewa Decyzyjne, SVM) z prostą siecią
    neuronową w problemie klasyfikacji autentyczności banknotów.
  - Zadanie 2 (CIFAR-10): Analizuje wpływ głębokości sieci neuronowej (mała vs duża architektura CNN)
    na skuteczność rozpoznawania obiektów na zdjęciach.
  - Zadanie 3 (Fashion MNIST): Wizualizuje skuteczność modelu klasyfikującego ubrania za pomocą
    Macierzy Pomyłek, pozwalając zidentyfikować najczęściej mylone klasy.
  - Zadanie 4 (Boston Housing): Przedstawia zastosowanie sieci neuronowych w problemie regresji
    (przewidywanie ciągłej wartości ceny mieszkania).

Autorzy:
  - Jakub Skarżyński
  - Sebastian Hellak

Instrukcja uruchomienia:
  1. Zainstaluj wymagane biblioteki:
       pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
  2. Uruchom skrypt:
       python main.py
"""

import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

tf.random.set_seed(42)
np.random.seed(42)


def task_banknotes_comparison():
    """
    Przeprowadza porównanie klasycznych metod ML z siecią neuronową na zbiorze Banknote Authentication.

    Funkcja realizuje:
    1. Pobranie i przetworzenie danych (podział, skalowanie).
    2. Trenowanie klasycznych modeli: Drzewo Decyzyjne (Entropia i Gini), SVM (kernel liniowy).
    3. Budowę i trenowanie prostej sieci neuronowej (MLP) w TensorFlow.
    4. Wyświetlenie tabelarycznego zestawienia dokładności (Accuracy) wszystkich modeli.
    5. Automatyczną ocenę, czy sieć neuronowa poradziła sobie lepiej od SVM.

    Args:
        Brak argumentów.

    Returns:
        None: Funkcja wypisuje wyniki bezpośrednio na konsolę.
    """
    print("\n")
    print(" Z1: Porównanie")
    print("\n")

    url = "https://raw.githubusercontent.com/AbhiRoy96/Banknote-Authentication-UCI-Dataset/master/bank_notes.csv"
    try:
        df = pd.read_csv(url)
        if 'variance' not in df.columns:
            df.columns = ["variance", "skewness", "curtosis", "entropy", "class"]
    except:
        print("Błąd wczytywania danych.")
        return

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    print("1. Trenowanie modeli klasycznych (Drzewa, SVM)")
    
    dt_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
    dt_entropy.fit(X_train, y_train)
    acc_dt_ent = accuracy_score(y_test, dt_entropy.predict(X_test))

    dt_gini = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)
    dt_gini.fit(X_train, y_train)
    acc_dt_gini = accuracy_score(y_test, dt_gini.predict(X_test))

    svm_lin = SVC(kernel='linear', random_state=42)
    svm_lin.fit(X_train_sc, y_train)
    acc_svm = accuracy_score(y_test, svm_lin.predict(X_test_sc))

    print("2. Trenowanie Sieci Neuronowej (TensorFlow)")
    
    model = models.Sequential([
        layers.Input(shape=(4,)),
        layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    history = model.fit(X_train_sc, y_train, epochs=20, batch_size=8, verbose=0)
    
    _, acc_nn = model.evaluate(X_test_sc, y_test, verbose=0)

    print("\n")
    print(f"| Model                   | Wynik   |")
    print(f"| Drzewo (Entropy)        | {acc_dt_ent:.4f}  |")
    print(f"| Drzewo (Gini)           | {acc_dt_gini:.4f}  |")
    print(f"| SVM (Linear)            | {acc_svm:.4f}  |")
    print(f"| Sieć Neuronowa (Keras)  | {acc_nn:.4f}  |")
    print("\n")
    
    if acc_nn >= acc_svm:
        print("Sieć neuronowa dorównała lub przebiła SVM")
    else:
        print("Klasyczny SVM poradził sobie lepiej")


def task_cifar10():
    """
    Porównuje dwie architektury CNN na zbiorze CIFAR-10.

    Funkcja:
    1. Pobiera zbiór zdjęć (zwierzęta, pojazdy) i normalizuje dane.
    2. Definiuje 'Małą Sieć' (1 warstwa konwolucyjna).
    3. Definiuje 'Dużą Sieć' (3 warstwy konwolucyjne, więcej filtrów).
    4. Trenuje oba modele przez zadaną liczbę epok.
    5. Generuje wykresy porównawcze historii uczenia (dokładność treningowa vs walidacyjna).

    Args:
        Brak argumentów.

    Returns:
        None: Funkcja wyświetla wykresy matplotlib.
    """
    print("\n")
    print("Z2: Zwierzęta CIFAR-10 (Mała vs Duża sieć)")
    print("\n")

    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    limit = 5000
    x_train, y_train = x_train[:limit], y_train[:limit]
    x_test, y_test = x_test[:1000], y_test[:1000]

    model_s = models.Sequential([
        layers.Conv2D(16, (3,3), activation='relu', input_shape=(32,32,3)),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ], name="Mala_Siec")

    model_l = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ], name="Duza_Siec")

    plt.figure(figsize=(12, 5))

    def plot_history(history, title, metric='accuracy'):
        """
        Pomocnicza funkcja do rysowania krzywych uczenia.

        Args:
            history (keras.callbacks.History): Obiekt historii zwracany przez model.fit().
            title (str): Tytuł wykresu (np. nazwa modelu).
            metric (str, optional): Nazwa metryki do wizualizacji (domyślnie 'accuracy').

        Returns:
            None: Funkcja dodaje elementy do aktywnego obiektu pyplot.
        """
        val_metric = f'val_{metric}'
        acc = history.history[metric]
        val_acc = history.history[val_metric]
        epochs = range(1, len(acc) + 1)
    
        plt.plot(epochs, acc, label='Trening')
        plt.plot(epochs, val_acc, label='Walidacja')
        plt.title("CIFAR-10 " + title)
        plt.xlabel('Epoki')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
    
    for i, model in enumerate([model_s, model_l]):
        print(f"Trenowanie: {model.name}...")
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        hist = model.fit(x_train, y_train, epochs=8, validation_data=(x_test, y_test), verbose=0)
        
        acc = hist.history['val_accuracy'][-1]
        print(f" Wynik {model.name}: {acc*100:.2f}%")
        
        plt.subplot(1, 2, i+1)
        plot_history(hist, f"{model.name} (Acc: {acc:.2f})")
    
    plt.show()




def task_fashion_mnist():
    """
    Trenuje sieć CNN na zbiorze ubrań (Fashion MNIST) i analizuje błędy.

    Głównym celem funkcji jest wygenerowanie i wyświetlenie Macierzy Pomyłek,
    która pokazuje, jak często model myli poszczególne kategorie ubrań).

    Args:
        Brak argumentów.

    Returns:
        None: Funkcja wyświetla mapę macierzy pomyłek.
    """
    print("\n")
    print("Z3: Ubrania (Confusion Matrix)")
    print("\n")

    (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

    class_names = ['T-shirt', 'Spodnie', 'Sweter', 'Sukienka', 'Płaszcz',
                   'Sandał', 'Koszula', 'Tenisówka', 'Torebka', 'Botki']

    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=3, verbose=0) 

    y_probs = model.predict(x_test)
    y_pred = np.argmax(y_probs, axis=1)

    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Macierz Pomyłek - Fashion MNIST')
    plt.ylabel('Prawda')
    plt.xlabel('Predykcja')
    plt.show()

def task_housing_regression():
    """
    Rozwiązuje problem regresji na zbiorze Boston Housing.

    Kroki:
    1. Standaryzacja danych wejściowych (średnia=0, odchylenie=1).
    2. Trening sieci regresyjnej.
    3. Ewaluacja błędu średniego bezwzględnego w dolarach.
    4. Wizualizacja korelacji między ceną przewidzianą a rzeczywistą na wykresie punktowym.

    Args:
        Brak argumentów.

    Returns:
        None: Funkcja wyświetla wykres i przykładowe predykcje w konsoli.
    """
    print("\n")
    print(" Z4: Przewidywanie cen mieszkań (Boston Housing)")
    print(" Typ problemu: REGRESJA")
    print("\n")

    (x_train, y_train), (x_test, y_test) = datasets.boston_housing.load_data()

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    history = model.fit(x_train, y_train, epochs=50, validation_split=0.2, verbose=0)

    mse, mae = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nŚredni błąd predykcji (MAE): ${mae * 1000:.2f}")

    predictions = model.predict(x_test).flatten()

    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, predictions, alpha=0.6)
    plt.plot([0, 50], [0, 50], 'r--') 
    plt.xlabel('Prawdziwa Cena ($1000s)')
    plt.ylabel('Przewidziana Cena ($1000s)')
    plt.title('Regresja: Ceny Mieszkań w Bostonie')
    plt.grid(True)
    
    print("\nPrzykładowe wyceny:")
    for i in range(3):
        print(f"Mieszkanie {i+1}: Prawdziwa cena: ${y_test[i]*1000:.0f} -> Model: ${predictions[i]*1000:.0f}")

    plt.show()

def main():
    """
    Główna funkcja sterująca wykonaniem wszystkich eksperymentów.
    """
    task_banknotes_comparison()
    task_cifar10()
    task_fashion_mnist()
    task_housing_regression()

if __name__ == "__main__":
    main()
