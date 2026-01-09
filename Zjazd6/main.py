"""
OPIS PROBLEMU:
Program realizuje analizę obrazu wideo w czasie rzeczywistym z wykorzystaniem biblioteki MediaPipe Tasks.
Jego celem jest detekcja sylwetki człowieka oraz rozpoznawanie specyficznych gestów dłoni w celu
określenia poziomu zagrożenia. System klasyfikuje widoczną osobę w trzech stanach:
1. "Unarmed" (Nieuzbrojony) - stan domyślny, osoba wykryta, brak specyficznych gestów (zielona ramka).
2. "Surrendering" (Poddający się) - wykryto uniesienie obu nadgarstków powyżej poziomu nosa (niebieska ramka).
3. "Armed" (Uzbrojony) - wykryto gest dłoni imitujący pistolet (wyprostowany kciuk i palec wskazujący,
   reszta palców zgięta). Powoduje to wyświetlenie czerwonej ramki oraz celownika na twarzy.

AUTORZY:
Jakub Skarżyński, Sebastian Hellak

INSTRUKCJA UŻYCIA:
1. Upewnij się, że posiadasz zainstalowane wymagane biblioteki:
   pip install opencv-python mediapipe
2. Uruchom skrypt w środowisku Python
3. Przy pierwszym uruchomieniu program automatycznie pobierze wymagane modele AI (.task)
   z serwerów Google. Wymagane jest połączenie z internetem.
4. Aby zakończyć działanie programu, naciśnij klawisz 'q' gdy aktywne jest okno wideo.
"""

import cv2
import mediapipe as mp
import math
import urllib.request

from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmark,
)
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarker,
    PoseLandmarkerOptions,
)
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core import image as mp_image

def is_finger_gun(hand_landmarks):
    """
    Analizuje punkty charakterystyczne dłoni, aby sprawdzić, czy ułożone są w gest pistoletu.

    Logika gestu:
    - Kciuk i palec wskazujący są wyprostowane (otwarte).
    - Palec środkowy, serdeczny i mały są zgięte (zamknięte).

    Args:
        hand_landmarks: Lista obiektów NormalizedLandmark z biblioteki MediaPipe,
                        reprezentująca punkty dłoni.

    Returns:
        bool: True, jeśli gest pistoletu został wykryty, w przeciwnym razie False.
    """
    # Indeksy punktów (TIP to czubek palca, PIP/MCP to stawy)
    thumb_tip = hand_landmarks[HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks[HandLandmark.THUMB_IP]

    index_tip = hand_landmarks[HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks[HandLandmark.INDEX_FINGER_PIP]

    middle_tip = hand_landmarks[HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks[HandLandmark.MIDDLE_FINGER_PIP]

    ring_tip = hand_landmarks[HandLandmark.RING_FINGER_TIP]
    ring_pip = hand_landmarks[HandLandmark.RING_FINGER_PIP]

    pinky_tip = hand_landmarks[HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks[HandLandmark.PINKY_PIP]

    def is_finger_open(tip, pip, wrist):
        """Pomocnicza funkcja sprawdzająca czy palec jest wyprostowany względem nadgarstka."""
        dist_tip = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
        dist_pip = math.hypot(pip.x - wrist.x, pip.y - wrist.y)
        return dist_tip > dist_pip

    wrist = hand_landmarks[HandLandmark.WRIST]

    index_open = is_finger_open(index_tip, index_pip, wrist)
    thumb_open = is_finger_open(thumb_tip, thumb_ip, wrist)
    middle_closed = not is_finger_open(middle_tip, middle_pip, wrist)
    ring_closed = not is_finger_open(ring_tip, ring_pip, wrist)
    pinky_closed = not is_finger_open(pinky_tip, pinky_pip, wrist)

    return index_open and thumb_open and middle_closed and ring_closed and pinky_closed


def get_bbox_from_pose(pose_landmarks, frame_w, frame_h):
    """
    Oblicza współrzędne prostokąta (bounding box) otaczającego wykrytą sylwetkę.

    Args:
        pose_landmarks: Lista znormalizowanych punktów sylwetki.
        frame_w (int): Szerokość klatki wideo.
        frame_h (int): Wysokość klatki wideo.

    Returns:
        tuple: (x1, y1, x2, y2) współrzędne lewego górnego i prawego dolnego rogu prostokąta
               z dodanym marginesem.
    """
    x_list = [lm.x for lm in pose_landmarks]
    y_list = [lm.y for lm in pose_landmarks]

    min_x, max_x = min(x_list), max(x_list)
    min_y, max_y = min(y_list), max(y_list)

    return (
        int(min_x * frame_w) - 20,
        int(min_y * frame_h) - 50,
        int(max_x * frame_w) + 20,
        int(max_y * frame_h) + 20,
    )


def draw_reticle(image, cx, cy, size=40, color=(0, 0, 255), thickness=2):
    """
    Rysuje celownik (okrąg z krzyżem) w zadanym punkcie.

    Args:
        image: Obraz OpenCV (numpy array), na którym rysujemy.
        cx (int): Współrzędna X środka celownika.
        cy (int): Współrzędna Y środka celownika.
        size (int, optional): Rozmiar celownika. Domyślnie 40.
        color (tuple, optional): Kolor w formacie BGR. Domyślnie czerwony.
        thickness (int, optional): Grubość linii. Domyślnie 2.
    """
    h, w = image.shape[:2]
    cv2.circle(image, (cx, cy), int(size * 0.25), color, thickness)
    cv2.line(image, (cx - size, cy), (cx + size, cy), color, thickness)
    cv2.line(image, (cx, cy - size), (cx, cy + size), color, thickness)

def main():
    """
    Główna funkcja programu.

    Otwiera kamerę, pobiera modele ML z internetu (zawsze), inicjalizuje MediaPipe
    Pose i Hand Landmarker, a następnie przetwarza klatki w pętli w celu wykrywania
    gestów i wyświetlania wyników na obrazie.

    Jeśli pobieranie modeli nie powiedzie się, funkcja wypisuje błąd i kończy działanie.
    """
    cap = cv2.VideoCapture(0) # 0 to domyślna kamera


    POSE_MODEL_PATH = 'pose_landmarker.task'
    HAND_MODEL_PATH = 'hand_landmarker.task'
    POSE_MODEL_URL = 'https://storage.googleapis.com/mediapipe-assets/pose_landmarker.task'
    HAND_MODEL_URL = 'https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task'

    def download_model(path: str, url: str) -> None:
        """Pobiera plik modelu z podanego URL i zapisuje go lokalnie pod podaną ścieżką.

        Args:
            path (str): Lokalna ścieżka, gdzie zostanie zapisany plik modelu.
            url (str): URL skąd pobrać model.

        Raises:
            Exception: Błędy związane z siecią lub zapisem pliku są przekazywane dalej po wypisaniu komunikatu.
        """
        print(f'Downloading {path} ...')
        urllib.request.urlretrieve(url, path)

    download_model(POSE_MODEL_PATH, POSE_MODEL_URL)
    download_model(HAND_MODEL_PATH, HAND_MODEL_URL)

    pose_options = PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH))
    hand_options = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH), num_hands=2)

    with PoseLandmarker.create_from_options(pose_options) as pose, \
         HandLandmarker.create_from_options(hand_options) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False

            mp_input = mp_image.Image(mp_image.ImageFormat.SRGB, image_rgb)
            pose_results = pose.detect(mp_input)
            hands_results = hands.detect(mp_input)
            
            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            h, w, _ = image.shape
            status = "Unarmed"
            color = (0, 255, 0)

            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks[0]
                
                # Indeksy: NOSE=0, LEFT_WRIST=15, RIGHT_WRIST=16, LEFT_EYE=2
                left_wrist_y = landmarks[15].y
                right_wrist_y = landmarks[16].y
                nose_y = landmarks[0].y
                eye_y = landmarks[2].y

                if left_wrist_y < nose_y and right_wrist_y < nose_y:
                    status = "Surrendering"
                    color = (255, 0, 0) # BGR
                
                elif hands_results.hand_landmarks:
                    for hand_landmarks in hands_results.hand_landmarks:
                        if is_finger_gun(hand_landmarks):
                            status = "Armed"
                            color = (0, 0, 255)
                            break

                if status == "Armed":
                    head = landmarks[0]  # nose
                    cx, cy = int(head.x * w), int(head.y * h)
                    draw_reticle(image, cx, cy, size=40, color=(0, 0, 255), thickness=2)

                x1, y1, x2, y2 = get_bbox_from_pose(landmarks, w, h)
                
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                
                cv2.rectangle(image, (x1, y1 - 40), (x1 + 250, y1), color, -1)
                
                cv2.putText(image, status, (x1 + 10, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('System Detekcji', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
