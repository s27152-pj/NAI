"""
Analiza Klasyfikacji Danych - Banknoty i Trzęsienia Ziemi

Opis problemu:
  - Aplikacja przeprowadza analizę porównawczą algorytmów klasyfikacji dla dwóch różnych zbiorów danych.
  - Część 1 (Banknoty): Rozwiązuje problem klasyfikacji binarnej (autentyczny/fałszywy).
    Porównuje skuteczność Drzew Decyzyjnych (z kryterium Entropii oraz Gini) oraz
    liniowego Maszyny Wektorów Nośnych (SVM).
  - Część 2 (Trzęsienia Ziemi): Rozwiązuje problem klasyfikacji wieloklasowej (poziomy alertów).
    Obejmuje wizualizację danych, skalowanie cech oraz optymalizację hiperparametrów
    SVM poprzez testowanie różnych funkcji jądra (Kernel functions).

Autorzy:
  - Jakub Skarżyński, Sebastian Hellak

Instrukcja uruchomienia:
  1. Upewnij się, że plik 'earthquake_alert_balanced_dataset.csv' znajduje się w folderze projektu.
  2. Zainstaluj wymagane biblioteki:
       pip install pandas numpy scikit-learn matplotlib seaborn
  3. Uruchom skrypt:
       python main.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Stałe globalne
URL_BANKNOTE = "https://raw.githubusercontent.com/AbhiRoy96/Banknote-Authentication-UCI-Dataset/master/bank_notes.csv"
FILE_EARTHQUAKE = "earthquake_alert_balanced_dataset.csv"

def run_banknote_analysis(url):
    """
    Przeprowadza proces trenowania i oceny modeli dla zbioru danych Banknote Authentication.

    Kroki:
    1. Pobranie danych z URL i nadanie nagłówków.
    2. Podział na zbiór treningowy i testowy (70/30).
    3. Standaryzacja danych (StandardScaler).
    4. Trenowanie trzech modeli: Drzewo (Entropia), Drzewo (Gini), SVM (Liniowy).
    5. Prezentacja wyników dokładności (Accuracy).
    6. Demonstracja działania na losowych próbkach.

    Args:
        url (str): Bezpośredni link do pliku CSV z danymi banknotów.

    Returns:
        None: Funkcja wypisuje wyniki bezpośrednio na konsolę.
    """
    print("--- CZĘŚĆ 1: Banknote Authentication ---")

    try:
        df_bn = pd.read_csv(url)
        if 'variance' not in df_bn.columns:
            df_bn.columns = ["variance", "skewness", "curtosis", "entropy", "class"]
        print("Dane Banknote załadowane pomyślnie.")
    except Exception as e:
        print(f"Błąd pobierania danych (możliwy brak sieci): {e}")
        return

    # Przygotowanie danych
    X_bn = df_bn.iloc[:, :-1]
    y_bn = df_bn.iloc[:, -1]
    X_train_bn, X_test_bn, y_train_bn, y_test_bn = train_test_split(X_bn, y_bn, test_size=0.3, random_state=42)

    scaler_bn = StandardScaler()
    X_train_bn_sc = scaler_bn.fit_transform(X_train_bn)
    X_test_bn_sc = scaler_bn.transform(X_test_bn)

    # Inicjalizacja modeli
    dt_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
    dt_gini = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)
    svm_bn = SVC(kernel='linear', random_state=42)

    # Trenowanie
    dt_entropy.fit(X_train_bn, y_train_bn)
    dt_gini.fit(X_train_bn, y_train_bn)
    svm_bn.fit(X_train_bn_sc, y_train_bn)

    # Ewaluacja
    print(f"Drzewo (Entropy, depth=4) Dokładność: {accuracy_score(y_test_bn, dt_entropy.predict(X_test_bn)):.4f}")
    print(f"Drzewo (Gini, depth=4)    Dokładność: {accuracy_score(y_test_bn, dt_gini.predict(X_test_bn)):.4f}")
    print(f"SVM (Linear)              Dokładność: {accuracy_score(y_test_bn, svm_bn.predict(X_test_bn_sc)):.4f}")

    # Przykładowa klasyfikacja
    print(f"\n--- Przykładowa klasyfikacja (Banknoty - 5 próbek) ---")
    if len(X_test_bn) > 0:
        idx = np.random.choice(X_test_bn.index, min(5, len(X_test_bn)), replace=False)
        for i in idx:
            row_df = X_test_bn.loc[[i]]
            true_val = y_test_bn.loc[i]
            features_scaled = scaler_bn.transform(row_df)
            pred_val = svm_bn.predict(features_scaled)[0]

            status_pred = "AUTENTYCZNY" if pred_val == 0 else "FAŁSZYWY"
            status_true = "AUTENTYCZNY" if true_val == 0 else "FAŁSZYWY"
            f = row_df.iloc[0].values

            print(f"Cechy: {np.round(f, 2)} -> Model: {status_pred} | Prawda: {status_true}")


def run_earthquake_analysis(filename):
    """
    Realizuje wizualizację i klasyfikację poziomów alertów trzęsień ziemi.

    Funkcja wykonuje:
    1. Wczytanie danych z pliku lokalnego.
    2. Generowanie wykresu rozrzutu (Scatterplot) Magnituda vs Intensywność.
    3. Kodowanie etykiet tekstowych (LabelEncoding) i skalowanie cech.
    4. Iteracyjne testowanie różnych jąder SVM (Linear, RBF, Poly, Sigmoid).
    5. Wybór najlepszego modelu i wyświetlenie pełnego raportu klasyfikacji.
    6. Prezentację predykcji wraz z pewnością (probability) dla losowych próbek.

    Args:
        filename (str): Nazwa pliku CSV z danymi sejsmicznymi.

    Returns:
        None: Funkcja generuje wykres (okno matplotlib) i wypisuje wyniki w konsoli.
    """
    print("\n--- CZĘŚĆ 2: Earthquake Alert Prediction ---")

    try:
        df_eq = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku {filename}")
        return

    # 1. Wizualizacja Danych
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    sns.scatterplot(
        data=df_eq,
        x='magnitude',
        y='mmi',
        hue='alert',
        style='alert',
        palette={'green': 'green', 'yellow': 'gold', 'orange': 'orange', 'red': 'red'},
        s=100,
        alpha=0.8
    )
    plt.title('Klasyfikacja Alertów Trzęsień Ziemi (Magnituda vs MMI)')
    plt.xlabel('Magnituda')
    plt.ylabel('Intensywność (MMI)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    print("Generuję wykres...")
    plt.show()

    # 2. Przygotowanie danych
    le = LabelEncoder()
    df_eq['alert_encoded'] = le.fit_transform(df_eq['alert'])

    features = ['magnitude', 'depth', 'cdi', 'mmi', 'sig']
    X_eq = df_eq[features]
    y_eq = df_eq['alert_encoded']

    X_train_eq, X_test_eq, y_train_eq, y_test_eq = train_test_split(X_eq, y_eq, test_size=0.3, random_state=42)

    scaler_eq = StandardScaler()
    X_train_eq_sc = scaler_eq.fit_transform(X_train_eq)
    X_test_eq_sc = scaler_eq.transform(X_test_eq)

    # 3. Testowanie konfiguracji SVM
    print("\nPorównanie konfiguracji SVM:")
    configs = [
        {'kernel': 'linear', 'C': 1.0},
        {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},
        {'kernel': 'rbf', 'C': 10.0, 'gamma': 0.1},
        {'kernel': 'poly', 'degree': 3, 'C': 1.0},
        {'kernel': 'sigmoid', 'C': 1.0}
    ]

    best_acc = 0
    best_model = None

    for conf in configs:
        svc = SVC(**conf, random_state=42, probability=True)
        svc.fit(X_train_eq_sc, y_train_eq)

        y_pred = svc.predict(X_test_eq_sc)
        acc = accuracy_score(y_test_eq, y_pred)

        params_str = ", ".join([f"{k}={v}" for k, v in conf.items()])
        print(f"SVM [{params_str}]: Accuracy = {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model = svc

    print(f"\nNajlepszy model: {best_model.kernel} (Acc: {best_acc:.4f})")
    y_pred_best = best_model.predict(X_test_eq_sc)

    print("\n--- Szczegółowy Raport Klasyfikacji ---")
    print(classification_report(y_test_eq, y_pred_best, target_names=le.classes_))

    # 4. Przykładowa klasyfikacja
    print(f"\n--- Przykładowa klasyfikacja (5 losowych próbek z danych testowych) ---")
    if len(X_test_eq) > 0:
        n_samples = min(5, len(X_test_eq))
        random_indices = np.random.choice(X_test_eq.index, n_samples, replace=False)

        for i in random_indices:
            row_data = X_test_eq.loc[[i]]
            true_idx = y_test_eq.loc[i]
            true_label = le.inverse_transform([true_idx])[0]

            sample_scaled = scaler_eq.transform(row_data)
            pred_idx = best_model.predict(sample_scaled)[0]
            pred_label = le.inverse_transform([pred_idx])[0]

            probs = best_model.predict_proba(sample_scaled)[0]
            confidence = np.max(probs)

            d = row_data.iloc[0]
            print(f"Dane: [Mag={d['magnitude']:.1f}, Depth={d['depth']:.0f}, CDI={d['cdi']:.1f}, MMI={d['mmi']:.1f}, Sig={d['sig']:.0f}]")
            print(f"      -> Model: {pred_label.upper()} ({confidence:.0%}) | Prawda: {true_label.upper()}")
            print("-" * 60)


def main():
    """
    Główna funkcja sterująca wykonaniem obu części analizy.
    """
    run_banknote_analysis(URL_BANKNOTE)
    run_earthquake_analysis(FILE_EARTHQUAKE)


if __name__ == "__main__":
    main()
