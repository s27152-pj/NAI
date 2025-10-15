"""
Zasady gry Hex:
- Gra planszowa dla dwóch graczy na planszy sześciokątnej.
- Gracze na przemian umieszczają swoje pionki (X i O) na wolnych polach planszy.
- Celem gracza X jest połączenie lewej i prawej krawędzi planszy.
- Celem gracza O jest połączenie górnej i dolnej krawędzi planszy.
- Pierwszy gracz, który połączy swoje krawędzie, wygrywa.
 
- link do zasad: https://pl.wikipedia.org/wiki/Hex_(gra_planszowa)
- Autorzy: Jakub Skarzyński, Sebastian Hellak

- instrukcja uruchomienia:
  1. Zainstaluj bibliotekę easyAI: pip install easyAI
    2. Uruchom ten skrypt: python game.py
- Sterowanie:
  - Gracz ludzki (X) wprowadza ruchy w formacie 'A5', 'B3' itd., gdzie litera oznacza kolumnę, a cyfra wiersz.
  - Komputer (O) wykonuje ruchy automatycznie.
"""

from easyAI import TwoPlayerGame, AI_Player, Human_Player, Negamax
import string
from collections import deque

N = 5

def shortest_path_length(board, player):
    """
    Oblicza najkrótszą ścieżkę gracza od krawędzi startowej do końcowej na planszy Hex.

    Algorytm BFS:
    - Dla gracza 'X' szuka połączenia od lewej do prawej krawędzi.
    - Dla gracza 'O' szuka połączenia od góry do dołu.

    Args:
        board (list[list[str]]): Aktualna plansza gry.
        player (str): Symbol gracza ('X' lub 'O').

    Returns:
        int: Długość najkrótszej ścieżki. Jeśli brak ścieżki, zwraca 99.
    """
    visited = set()
    queue = deque()
    N = len(board)
    # Dodaj do kolejki wszystkie pola startowe gracza
    if player == 'X':
        for i in range(N):
            if board[i][0] == 'X':
                queue.append((i, 0, 0))  # (wiersz, kolumna, dystans)
    else:
        for j in range(N):
            if board[0][j] == 'O':
                queue.append((0, j, 0))
    # BFS: przeszukuje planszę
    while queue:
        x, y, dist = queue.popleft()
        if (x, y) in visited:
            continue
        visited.add((x, y))
        # Sprawdź czy osiągnięto krawędź końcową
        if player == 'X' and y == N - 1:
            return dist
        if player == 'O' and x == N - 1:
            return dist
        # Dodaj sąsiadów gracza do kolejki
        for dx, dy in [(-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < N and board[nx][ny] == player and (nx, ny) not in visited:
                queue.append((nx, ny, dist + 1))
    return 99  # Brak ścieżki

class HexGame(TwoPlayerGame):
    """
    Klasa gry Hex dla easyAI.

    Obsługuje logikę gry, wyświetlanie planszy, wykonywanie ruchów,
    sprawdzanie zwycięstwa oraz funkcję oceny dla AI.
    """
    def __init__(self, players):
        """
        Inicjalizuje nową grę Hex.

        Args:
            players (list): Lista graczy (Human_Player, AI_Player).
        """
        self.players = players
        self.board = [['.' for _ in range(N)] for _ in range(N)]
        self.current_player = 1  # 1: człowiek (X), 2: komputer (O)

    def possible_moves(self):
        """
        Zwraca listę możliwych ruchów w formacie 'A5', 'B3' itd.

        Returns:
            list[str]: Lista dostępnych ruchów.
        """
        moves = []
        for r in range(N):
            for c in range(N):
                if self.board[r][c] == '.':
                    moves.append(f"{string.ascii_uppercase[c]}{r+1}")
        return moves

    def make_move(self, move):
        """
        Wykonuje ruch na planszy.

        Args:
            move (str): Ruch w formacie 'A5', 'B3' itd.
        """
        col = string.ascii_uppercase.find(move[0])
        row = int(move[1:]) - 1
        self.board[row][col] = self.player_symbol()

    def unmake_move(self, move):
        """
        Cofnięcie ruchu (potrzebne dla AI podczas przeszukiwania).

        Args:
            move (str): Ruch do cofnięcia.
        """
        col = string.ascii_uppercase.find(move[0])
        row = int(move[1:]) - 1
        self.board[row][col] = '.'

    def show(self):
        """
        Wyświetla aktualną planszę gry w konsoli.
        """
        print("   " + " ".join(string.ascii_uppercase[:N]))
        for i in range(N):
            print(" " * i + f"{i+1:2} " + " ".join(self.board[i]))
        print()

    def player_symbol(self):
        """
        Zwraca symbol obecnego gracza.

        Returns:
            str: 'X' dla człowieka, 'O' dla komputera.
        """
        return 'X' if self.current_player == 1 else 'O'

    def get_neighbors(self, x, y):
        """
        Generator współrzędnych sąsiadów pola (x, y) na planszy Hex.

        Args:
            x (int): Wiersz.
            y (int): Kolumna.

        Yields:
            tuple[int, int]: Współrzędne sąsiada.
        """
        dirs = [(-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0)]
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < N:
                yield nx, ny

    def check_win(self, player):
        """
        Sprawdza czy podany gracz wygrał (czy połączył swoje krawędzie).

        Args:
            player (str): Symbol gracza ('X' lub 'O').

        Returns:
            bool: True jeśli gracz wygrał, False w przeciwnym razie.
        """
        visited = set()
        stack = []
        # Dodaj do stosu wszystkie pola startowe gracza
        if player == 'X':
            for i in range(N):
                if self.board[i][0] == 'X':
                    stack.append((i, 0))
        else:
            for j in range(N):
                if self.board[0][j] == 'O':
                    stack.append((0, j))
        # DFS: przeszukuje planszę
        while stack:
            x, y = stack.pop()
            if (x, y) in visited: 
                continue
            visited.add((x, y))
            # Sprawdź czy osiągnięto krawędź końcową
            if player == 'X' and y == N - 1:
                return True
            if player == 'O' and x == N - 1:
                return True
            # Dodaj sąsiadów gracza do stosu
            for nx, ny in self.get_neighbors(x, y):
                if self.board[nx][ny] == player and (nx, ny) not in visited:
                    stack.append((nx, ny))
        return False

    def win(self):
        """
        Sprawdza czy obecny gracz wygrał.

        Returns:
            bool: True jeśli obecny gracz wygrał.
        """
        return self.check_win(self.player_symbol())

    def is_over(self):
        """
        Sprawdza czy gra się zakończyła (wygrana lub brak ruchów).

        Returns:
            bool: True jeśli gra się zakończyła.
        """
        return self.win() or all(cell != '.' for row in self.board for cell in row)

    def scoring(self):
        """
        Funkcja oceny planszy dla AI.

        Im krótsza ścieżka do zwycięstwa dla AI, tym lepiej.
        AI minimalizuje swoją odległość do wygranej i maksymalizuje odległość przeciwnika.

        Returns:
            int: Wynik oceny planszy.
        """
        me = self.player_symbol()
        opp = 'O' if me == 'X' else 'X'
        my_path = shortest_path_length(self.board, me)
        opp_path = shortest_path_length(self.board, opp)
        return opp_path - my_path

if __name__ == "__main__":
    # Uruchamia grę Hex: człowiek kontra AI
    ai_algo = Negamax(4)  # Głębokość przeszukiwania AI (większa = lepsza, ale wolniejsza)
    game = HexGame([Human_Player(), AI_Player(ai_algo)])
    game.play()
    if game.win():
        print(f"Gracz {game.player_symbol()} wygrywa!")

