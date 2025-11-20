"""
Silnik Rekomendacji Filmów - K-Means Clustering

Opis problemu:
  - Aplikacja implementuje system rekomendacji filmów i seriali oparty na uczeniu maszynowym.
  - Wykorzystuje algorytm K-Means do grupowania
    użytkowników o podobnych gustach na podstawie ich historycznych ocen.
  - Program przetwarza dane z pliku CSV, tworzy macierz użytkownik-film, dzieli populację
    na klastry i generuje rekomendacje (oraz anty-rekomendacje) na podstawie średnich ocen grupy.
  - Dodatkowo aplikacja pobiera szczegóły filmów z API OMDb.

Autorzy:
  - Jakub Skarżyński
  - Sebastian Hellak

Instrukcja uruchomienia:
  1. Upewnij się, że plik 'oceny_clean.csv' znajduje się w folderze projektu.
  2. Zainstaluj wymagane biblioteki:
       pip install pandas numpy scikit-learn requests
  3. Uruchom skrypt:
       python main.py
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import requests

def load_data(filename):
    """
        Wczytuje surowe dane z pliku CSV i przetwarza je na macierz użytkownik-film.

        Funkcja oczekuje pliku bez nagłówków, gdzie dane ułożone są w formacie:
        [Użytkownik]; [Film1]; [Ocena1]; [Film2]; [Ocena2]...

        Jeśli wiersze mają różną długość, funkcja dynamicznie je przetwarza.

        Args:
            filename (str): Ścieżka do pliku CSV.

        Returns:
            pandas.DataFrame: Macierz (Pivot Table), gdzie:
                - Indeks (wiersze) = Nazwy użytkowników
                - Kolumny = Tytuły filmów
                - Wartości = Oceny (0.0 - 10.0).
                Zwraca None w przypadku błędu odczytu pliku.
        """
    df = pd.read_csv(filename, sep=';', encoding='utf-8-sig', header=None)

    data_list = []
    for idx, row in df.iterrows():
        user = row.iloc[0]
        for i in range(1, len(df.columns) - 1, 2):
            film = row.iloc[i]
            score = row.iloc[i + 1]
            if pd.notna(film) and pd.notna(score):
                data_list.append({
                    'user': user,
                    'title': str(film).strip(),
                    'rating': float(score)
                })

    df_long = pd.DataFrame(data_list)
    matrix = df_long.pivot_table(index='user', columns='title', values='rating').fillna(0)

    return matrix

def get_kmeans_recommendations(user_name, matrix, n_clusters=3):
    """
        Generuje rekomendacje przy użyciu algorytmu K-Means.

        Algorytm działa w 4 krokach:
        1. Grupuje wszystkich użytkowników na N klastrów (grup) o podobnych gustach.
        2. Identyfikuje, do której grupy trafił wybrany użytkownik.
        3. Oblicza średnią ocenę dla każdego filmu wewnątrz tej grupy.
        4. Rekomenduje filmy z najwyższą średnią w grupie, których użytkownik jeszcze nie widział.

        Args:
            user_name (str): Nazwa użytkownika.
            matrix (pandas.DataFrame): Macierz ocen użytkowników.
            n_clusters (int, optional): Liczba grup, na które dzielimy populację.

        Returns:
            tuple: (recs, antis, error)
                - recs (pandas.Series): 5 filmów z najwyższą prognozowaną oceną.
                - antis (pandas.Series): 5 filmów z najniższą prognozowaną oceną.
                - error (str lub None): Komunikat błędu, jeśli użytkownik nie istnieje.
        """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(matrix)

    user_clusters = pd.Series(clusters, index=matrix.index, name='cluster')

    if user_name not in matrix.index:
        return None, None, f"Błąd: Użytkownik {user_name} nie istnieje."

    my_cluster_id = user_clusters[user_name]
    print(f" Użytkownik '{user_name}' trafił do Grupy nr {my_cluster_id}")

    users_in_my_cluster = user_clusters[user_clusters == my_cluster_id].index
    print(f" Liczba osób w tej grupie: {len(users_in_my_cluster)}")

    # Liczymy średnie dla klastra
    cluster_matrix = matrix.loc[users_in_my_cluster]
    mean_ratings = cluster_matrix.mean(axis=0)

    # Usuwamy to co user już widział
    user_seen = matrix.loc[user_name]
    seen_titles = user_seen[user_seen > 0].index

    recs = mean_ratings.drop(seen_titles).sort_values(ascending=False).head(5)

    viewed_by_cluster = mean_ratings[mean_ratings > 0]
    antis = viewed_by_cluster.drop(seen_titles).sort_values(ascending=True).head(5)

    return recs, antis, None

def get_movie_details(movie_title):
    """
        Pobiera metadane o filmie z zewnętrznego API (OMDb).

        Args:
            movie_title (str): Tytuł filmu w języku angielskim lub oryginalnym.

        Returns:
            str: Sformatowany tekst zawierający Rok, Gatunek i Opis fabuły.
                 W przypadku błędu lub braku filmu zwraca odpowiedni komunikat.
        """
    url = f"http://www.omdbapi.com/?t={movie_title}&apikey=bc3c88d4&plot=short"
    response = requests.get(url, timeout=5)
    data = response.json()

    if data.get('Response') == 'True':
        year = data.get('Year', 'N/A')
        genre = data.get('Genre', 'N/A')
        plot = data.get('Plot', 'Brak opisu.')
        if len(plot) > 100:
            plot = plot[:200] + "..."
        return f"\n  Info: {year} | {genre}\n  Opis: {plot}"
    else:
        return "\n    Info: Nie znaleziono w bazie OMDb"


def main():
    """
        Główna funkcja sterująca programem.

        Realizuje proces:
        1. Wczytanie danych z pliku.
        2. Wyświetlenie listy dostępnych użytkowników z przypisanymi ID.
        3. Pobranie od użytkownika numeru ID.
        4. Uruchomienie silnika rekomendacji dla wybranego profilu.
        5. Wyświetlenie wyników (Rekomendacje i Anty-rekomendacje) wraz z opisami.
    """
    matrix = load_data("oceny_clean.csv")
    if matrix is None: return

    users = matrix.index.tolist()

    print("\n LISTA UŻYTKOWNIKÓW:")
    print("=" * 30)
    for idx, user in enumerate(users):
        print(f"[{idx}] {user}")
    print("=" * 30)

    while True:
        try:
            user_input = input(f"\nWybierz ID użytkownika (0-{len(users) - 1}): ")
            user_id = int(user_input)
            if 0 <= user_id < len(users):
                target_user = users[user_id]
                break
            else:
                print(f"Podaj liczbę z zakresu 0-{len(users) - 1}")
        except ValueError:
            print("To nie jest liczba!")

    print(f"\nRekomendacje dla: {target_user}...")
    recs, antis, error = get_kmeans_recommendations(target_user, matrix, n_clusters=3)
    if error:
        print(error)
    else:
        print("\n" + "=" * 60)
        print(f"5 REKOMENDACJI:")
        print("=" * 60)
        for title, score in recs.items():
            info = get_movie_details(title)
            print(f" {title.upper()} (Prognoza: {score:.2f}/10) {info}")
            print("-" * 60)
        print("\n" + "=" * 60)
        print(f"ANTY-REKOMENDACJE:")
        print("=" * 60)
        for title, score in antis.items():
            info = get_movie_details(title)
            print(f" {title} (Prognoza: {score:.2f}/10) {info}")
            print("-" * 60)
          
if __name__ == "__main__":
    main()
