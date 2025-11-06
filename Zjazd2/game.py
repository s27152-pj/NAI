"""
Asystent parkowania - logika rozmyta

Opis problemu:
  - Symulacja asystenta parkowania wykorzystująca logikę rozmytą (scikit-fuzzy).
  - Aplikacja rysuje prosty model samochodu i miejsce parkingowe, oblicza
    sugerowany kąt skrętu na podstawie odległości czujników bocznych i tylnego.
  - Celem jest demonstracja reguł rozmytych i prostego sterowania kinematycznego.

Autorzy:
  - Jakub Skarżyński
  - Sebastian Hellak

- instrukcja uruchomienia:
  1. Zainstaluj biblioteki: 
        pip install scipy scikit-fuzzy customtkinter
  2. Uruchom ten skrypt: python game.py
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import customtkinter as ctk

# --- Stałe globalne ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
PPM = 40

CAR_WIDTH_M = 1.8
CAR_LENGTH_M = 4.5
PARKING_SPOT_WIDTH_M = 3.0
PARKING_SPOT_DEPTH_M = 5.0

# Obliczenia pozycji miejsca parkingowego
SPOT_CENTER_X_M = (SCREEN_WIDTH / 2) / PPM
SPOT_CENTER_Y_M = (SCREEN_HEIGHT - PARKING_SPOT_DEPTH_M * PPM / 2 - 20) / PPM
SPOT_TARGET_ANGLE_DEG = 90.0

# Pozycje startowe samochodu
START_X_M = SPOT_CENTER_X_M + 2.5
START_Y_M = SPOT_CENTER_Y_M - 7.0
START_ANGLE_DEG = 30

# Parametry symulacji
IDEAL_ANGLE_ERROR = 3.0
IDEAL_SIDE_ERROR_M = 0.25
REVERSE_STOP_DIST_M = 0.15
FORWARD_CORRECT_DIST_M = 3.0
SIM_TIMESTEP_S = 0.03


class FuzzyParkingAssistant:
    """
    Logika rozmyta dla sterowania kątem skrętu podczas cofania.

    Atrybuty:
        dist_left, dist_right, dist_back : Antecedent
            Wejścia (odległości) przeskalowane do zakresów używanych w regułach.
        steering_angle : Consequent
            Wyjście - sugerowany kąt skrętu w stopniach.
        rules : list[ctrl.Rule]
            Lista reguł rozmytych definiujących zachowanie systemu.
    """

    def __init__(self):
        # Wejścia
        self.dist_left = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'dist_left')
        self.dist_right = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'dist_right')
        self.dist_back = ctrl.Antecedent(np.arange(0, 5.1, 0.1), 'dist_back')

        # Wyjście
        self.steering_angle = ctrl.Consequent(np.arange(-45, 46, 1), 'steering_angle')

        self._define_membership_functions()
        self.rules = self._define_rules()

        self.steering_control = ctrl.ControlSystem(self.rules)
        self.steering_simulation = ctrl.ControlSystemSimulation(self.steering_control)

    def _define_membership_functions(self):
        """
        Definiuje funkcje przynależności dla wejść i wyjścia.

        Funkcje przynależności są tworzone dla sensorów bocznych i tylnego oraz
        dla wyjściowego kąta skrętu.
        """
        for sensor in [self.dist_left, self.dist_right]:
            sensor['too_close'] = fuzz.trapmf(sensor.universe, [0, 0, 0.1, 0.2])
            sensor['close'] = fuzz.trimf(sensor.universe, [0.15, 0.35, 0.55])
            sensor['ok'] = fuzz.trimf(sensor.universe, [0.5, 0.6, 0.7])
            sensor['far'] = fuzz.trapmf(sensor.universe, [0.65, 0.8, 1.0, 1.0])

        self.dist_back['very_close'] = fuzz.trimf(self.dist_back.universe, [0, 0, 0.5])
        self.dist_back['close'] = fuzz.trimf(self.dist_back.universe, [0.3, 1.0, 1.7])
        self.dist_back['ok'] = fuzz.trapmf(self.dist_back.universe, [1.5, 4.0, 5.0, 5.0])

        self.steering_angle['hard_left'] = fuzz.trimf(self.steering_angle.universe, [-45, -45, -30])
        self.steering_angle['left'] = fuzz.trimf(self.steering_angle.universe, [-35, -20, -5])
        self.steering_angle['straight'] = fuzz.trimf(self.steering_angle.universe, [-10, 0, 10])
        self.steering_angle['right'] = fuzz.trimf(self.steering_angle.universe, [5, 20, 35])
        self.steering_angle['hard_right'] = fuzz.trimf(self.steering_angle.universe, [30, 45, 45])

    def _define_rules(self):
        """
        Tworzy i zwraca listę reguł rozmytych.

        Zwraca:
            list: lista obiektów ctrl.Rule używanych w ControlSystem.
        """
        rule1 = ctrl.Rule(self.dist_left['too_close'], self.steering_angle['hard_right'])
        rule2 = ctrl.Rule(self.dist_right['too_close'], self.steering_angle['hard_left'])
        rule3 = ctrl.Rule(self.dist_left['close'] & self.dist_right['ok'], self.steering_angle['left'])
        rule4 = ctrl.Rule(self.dist_left['ok'] & self.dist_right['close'], self.steering_angle['right'])
        rule5 = ctrl.Rule(self.dist_left['ok'] & self.dist_right['ok'], self.steering_angle['straight'])
        rule6 = ctrl.Rule(self.dist_left['far'] & self.dist_right['close'], self.steering_angle['right'])
        rule7 = ctrl.Rule(self.dist_left['close'] & self.dist_right['far'], self.steering_angle['left'])
        rule8 = ctrl.Rule(self.dist_back['very_close'], self.steering_angle['straight'])
        rule9_new = ctrl.Rule(self.dist_left['far'], self.steering_angle['right'])
        rule10_new = ctrl.Rule(self.dist_right['far'], self.steering_angle['left'])

        return [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9_new, rule10_new]

    def compute_steering(self, left_dist, right_dist, back_dist):
        """
        Oblicza wyjściowy kąt skrętu na podstawie aktualnych odległości.

        Args:
            left_dist (float): odległość od lewej krawędzi w metrach (skalowana).
            right_dist (float): odległość od prawej krawędzi w metrach (skalowana).
            back_dist (float): odległość tyłu samochodu od tylnej linii miejsca (m).

        Returns:
            float: wyliczony kąt skrętu w stopniach (wartość numeryczna).
        """
        self.steering_simulation.input['dist_left'] = left_dist
        self.steering_simulation.input['dist_right'] = right_dist
        self.steering_simulation.input['dist_back'] = back_dist

        try:
            self.steering_simulation.compute()
            return self.steering_simulation.output['steering_angle']
        except ValueError:
            return 0.0


class Car:
    """
    Prostym model ruchu samochodu (kinematyka pojazdu).

    Atrybuty:
        x_m, y_m (float): pozycja środka pojazdu w metrach (układ ekranu -> piksele = *PPM).
        angle_deg (float): orientacja pojazdu w stopniach (0..360, 90 = dół ekranu itp.).
        speed_mps (float): prędkość w m/s (wartości ujemne = cofanie).
        steering_angle_deg (float): aktualny kąt skrętu kół (deg).
    """

    def __init__(self, x_m, y_m, angle_deg, length_m, width_m):
        self.length = length_m
        self.width = width_m
        self.reset(x_m, y_m, angle_deg)

    def reset(self, x_m, y_m, angle_deg):
        """
        Resetuje pozycję i stan pojazdu.

        Args:
            x_m (float): pozycja X w metrach
            y_m (float): pozycja Y w metrach
            angle_deg (float): kąt orientacji w stopniach
        """
        self.x_m = x_m
        self.y_m = y_m
        self.angle_deg = angle_deg
        self.speed_mps = 0.0
        self.steering_angle_deg = 0.0

    def update_physics(self, dt_s):
        """
        Aktualizuje pozycję i orientację pojazdu w oparciu o prosty model kinematyczny.

        Args:
            dt_s (float): krok czasu w sekundach
        """
        if self.speed_mps == 0:
            return

        steering_rad = math.radians(self.steering_angle_deg)
        angular_velocity_rad_s = (self.speed_mps * math.tan(steering_rad)) / self.length

        self.angle_deg += math.degrees(angular_velocity_rad_s) * dt_s
        self.angle_deg %= 360  # Normalizuj kąt

        angle_rad_world = math.radians(-self.angle_deg)
        velocity_x_mps = self.speed_mps * math.cos(angle_rad_world)
        velocity_y_mps = self.speed_mps * math.sin(angle_rad_world)

        self.x_m += velocity_x_mps * dt_s
        self.y_m += velocity_y_mps * dt_s

    def get_corners_px(self):
        """
        Zwraca współrzędne narożników prostokąta pojazdu w pikselach.

        Returns:
            dict: mapowanie nazw narożników ('fl','fr','rl','rr') -> (x_px, y_px)
        """
        x_px, y_px = self.x_m * PPM, self.y_m * PPM
        angle_rad_tk = math.radians(-self.angle_deg)

        w_half_px = self.width * PPM / 2
        l_half_px = self.length * PPM / 2

        corners_local = {
            'fl': (l_half_px, -w_half_px),
            'fr': (l_half_px, w_half_px),
            'rl': (-l_half_px, -w_half_px),
            'rr': (-l_half_px, w_half_px)
        }

        corners_world_px = {}
        for name, (lx, ly) in corners_local.items():
            x_rot = lx * math.cos(angle_rad_tk) - ly * math.sin(angle_rad_tk)
            y_rot = lx * math.sin(angle_rad_tk) + ly * math.cos(angle_rad_tk)
            corners_world_px[name] = (x_px + x_rot, y_px + y_rot)

        return corners_world_px


class ParkingApp:
    """
    Aplikacja GUI zarządzająca symulacją parkowania.

    Metody odpowiadają za przygotowanie GUI, pętlę symulacji, aktualizację stanu i rysowanie.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Asystent Parkowania - Logika Rozmyta (Wersja OOP)")

        self.assistant = FuzzyParkingAssistant()
        self.car = Car(START_X_M, START_Y_M, START_ANGLE_DEG, CAR_LENGTH_M, CAR_WIDTH_M)
        self.state = 'PODJAZD'

        # Definicje miejsca parkingowego
        spot_pos_px = (SPOT_CENTER_X_M * PPM, SPOT_CENTER_Y_M * PPM)
        spot_width_px = PARKING_SPOT_WIDTH_M * PPM
        spot_depth_px = PARKING_SPOT_DEPTH_M * PPM
        self.spot_line_left_x = spot_pos_px[0] - spot_width_px / 2
        self.spot_line_right_x = spot_pos_px[0] + spot_width_px / 2
        self.spot_line_back_y = spot_pos_px[1] + spot_depth_px / 2

        self.setup_gui()

        self.root.bind('<r>', self.reset_simulation)
        self.root.bind('<R>', self.reset_simulation)

    def setup_gui(self):
        """
        Tworzy i rozmieszcza widgety GUI (informacje + canvas).

        Używa customtkinter do tworzenia ramek, etykiet oraz płótna symulacji.
        """
        # Ramka informacyjna
        self.info_frame = ctk.CTkFrame(self.root, width=250)
        self.info_frame.pack(side=ctk.LEFT, fill=ctk.Y, padx=10, pady=10)

        ctk.CTkLabel(self.info_frame, text="--- WEJŚCIA LOGIKI ROZMYTEJ ---", font=("Arial", 12, "bold")).pack(
            anchor="w")

        self.var_dist_left = ctk.StringVar(value="1. Czujnik Lewy (m): ...")
        ctk.CTkLabel(self.info_frame, textvariable=self.var_dist_left).pack(anchor="w")
        self.var_dist_right = ctk.StringVar(value="2. Czujnik Prawy (m): ...")
        ctk.CTkLabel(self.info_frame, textvariable=self.var_dist_right).pack(anchor="w")
        self.var_dist_back = ctk.StringVar(value="3. Czujnik Tył (m):   ...")
        ctk.CTkLabel(self.info_frame, textvariable=self.var_dist_back).pack(anchor="w")

        ctk.CTkLabel(self.info_frame, text="--- WYJŚCIE LOGIKI ROZMYTEJ ---", font=("Arial", 12, "bold")).pack(
            anchor="w", pady=(10, 0))
        self.var_steering = ctk.StringVar(value="Sugerowany skręt (stopnie): ...")
        ctk.CTkLabel(self.info_frame, textvariable=self.var_steering, text_color="green",
                     font=("Arial", 11, "bold")).pack(anchor="w")

        ctk.CTkLabel(self.info_frame, text="--- STAN SYSTEMU ---", font=("Arial", 12, "bold")).pack(anchor="w",
                                                                                                    pady=(10, 0))
        self.var_state = ctk.StringVar(value="...")
        self.label_state = ctk.CTkLabel(self.info_frame, textvariable=self.var_state, text_color="cyan",
                                        font=("Arial", 12, "bold"))
        self.label_state.pack(anchor="w")

        ctk.CTkLabel(self.info_frame, text="\nNaciśnij [R] aby zresetować", font=("Arial", 10)).pack(anchor="w",
                                                                                                     pady=20)

        # Płótno symulacji
        self.canvas = ctk.CTkCanvas(self.root, width=SCREEN_WIDTH, height=SCREEN_HEIGHT, bg="#333333")
        self.canvas.pack(side=ctk.RIGHT)

        # Rysowanie miejsca parkingowego
        l, r, b = self.spot_line_left_x, self.spot_line_right_x, self.spot_line_back_y
        t = b - PARKING_SPOT_DEPTH_M * PPM
        self.canvas.create_line(l, t, l, b, fill="yellow", width=3)
        self.canvas.create_line(r, t, r, b, fill="yellow", width=3)
        self.canvas.create_line(l, b, r, b, fill="yellow", width=3)

        # Rysowanie samochodu
        initial_corners = self.car.get_corners_px()
        corner_list_init = [coord for name in ['fl', 'fr', 'rr', 'rl'] for coord in initial_corners[name]]
        self.canvas.create_polygon(corner_list_init, fill="red", outline="black", tags="car")

    def reset_simulation(self, event=None):
        """
        Resetuje symulację do stanu początkowego i uruchamia pętlę aktualizacji.
        """
        self.car.reset(START_X_M, START_Y_M, START_ANGLE_DEG)
        self.state = 'PODJAZD'
        self.update_loop()

    def update_loop(self):
        """
        Główna pętla symulacji:
          1) Odczyt pozycji narożników pojazdu
          2) Obliczenie odległości sensorów (lewy, prawy, tył)
          3) Wyznaczenie stanu maszyny stanów i obliczenie sterowania
          4) Aktualizacja fizyki pojazdu
          5) Rysowanie na canvasie i odświeżenie wartości tekstowych
          6) Zaplanowanie kolejnego wywołania metodą root.after, jeśli symulacja nadal aktywna
        """
        corners_px = self.car.get_corners_px()

        dist_left_px = corners_px['rl'][0] - self.spot_line_left_x
        dist_left_m = np.clip(dist_left_px / PPM, 0.01, 1.0)

        dist_right_px = self.spot_line_right_x - corners_px['rr'][0]
        dist_right_m = np.clip(dist_right_px / PPM, 0.01, 1.0)

        car_back_y_px = max(corners_px['rl'][1], corners_px['rr'][1])
        dist_back_px = self.spot_line_back_y - car_back_y_px
        dist_back_m = np.clip(dist_back_px / PPM, 0.01, 5.0)

        # SPRAWDZANIE WARUNKÓW KRAŃCOWYCH
        angle_err_raw = self.car.angle_deg - SPOT_TARGET_ANGLE_DEG
        angle_err_raw = (angle_err_raw + 180) % 360 - 180
        side_balance_error_m = dist_left_m - dist_right_m

        is_parked_ok = (abs(angle_err_raw) < IDEAL_ANGLE_ERROR) and \
                       (abs(side_balance_error_m) < IDEAL_SIDE_ERROR_M)

        # MASZYNA STANÓW
        steering_output = 0.0

        if self.state == 'PODJAZD':
            self.label_state.configure(text_color="orange")
            self.var_state.set("STAN: Podjazd (Celowanie)")
            self.car.speed_mps = -1.0 


            # Cel 1: Ustawić środek samochodu (x_m) nad środkiem miejsca. Błąd > 0 (na prawo od celu) -> skręt w prawo
            error_x_m = self.car.x_m - SPOT_CENTER_X_M
            Kp_side = 2.0
            steering_from_side = -error_x_m * Kp_side

            # Cel 2: Ustawić kąt na 90 stopn. Błąd < 0 (np. 65 stopni) -> skręt w prawo, aby zwiększyć kąt
            Kp_angle = 10.0
            steering_from_angle = angle_err_raw * Kp_angle

            steering_output = steering_from_side + steering_from_angle

            # Warunek przejścia: Gdy tył jest wystarczająco blisko, przekaż kontrolę logice rozmytej
            if dist_back_m <= FORWARD_CORRECT_DIST_M:
                self.state = 'COFANIE'

        if self.state == 'COFANIE':
            self.label_state.configure(text_color="cyan")
            self.var_state.set("STAN: Cofanie")
            self.car.speed_mps = -1.0

            steering_output = self.assistant.compute_steering(dist_left_m, dist_right_m, dist_back_m)

            if dist_back_m <= REVERSE_STOP_DIST_M:
                self.state = 'ZAPARKOWANY' if is_parked_ok else 'KORYGOWANIE'

        elif self.state == 'KORYGOWANIE':
            self.label_state.configure(text_color="yellow")
            self.var_state.set("STAN: Korygowanie")
            self.car.speed_mps = 1.0

            Kp_angle = 2.0
            steering_from_angle = -angle_err_raw * Kp_angle
            Kp_side = 10.0
            steering_from_side = -side_balance_error_m * Kp_side
            steering_output = steering_from_angle + steering_from_side

            if dist_back_m > FORWARD_CORRECT_DIST_M:
                self.state = 'COFANIE'

        elif self.state == 'ZAPARKOWANY':
            self.label_state.configure(text_color="green")
            self.var_state.set("STAN: Zaparkowany idealnie!")
            self.car.speed_mps = 0
            steering_output = 0

        # AKTUALIZACJA FIZYKI SAMOCHODU
        self.car.steering_angle_deg = np.clip(steering_output, -45, 45)
        self.car.update_physics(SIM_TIMESTEP_S)

        # AKTUALIZACJA GUI
        corners_for_draw = self.car.get_corners_px()
        corner_list = [coord for name in ['fl', 'fr', 'rr', 'rl'] for coord in corners_for_draw[name]]
        self.canvas.coords("car", *corner_list)

        self.var_dist_left.set(f"1. Czujnik Lewy (m): {dist_left_m:.2f}")
        self.var_dist_right.set(f"2. Czujnik Prawy (m): {dist_right_m:.2f}")
        self.var_dist_back.set(f"3. Czujnik Tył (m):   {dist_back_m:.2f}")
        self.var_steering.set(f"Sugerowany skręt (stopnie): {steering_output:.2f}")

        # KONTYNUACJA PĘTLI
        if self.state not in ['ZAPARKOWANY']:
            self.root.after(int(SIM_TIMESTEP_S * 1000), self.update_loop)

    def run(self):
        """Uruchamia aplikację (reset + mainloop)."""
        self.reset_simulation()
        self.root.mainloop()


if __name__ == "__main__":
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    app = ParkingApp(root)
    app.run()
