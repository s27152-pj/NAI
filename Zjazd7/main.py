"""
OPIS PROBLEMU:
Program implementuje proces uczenia ze wzmocnieniem przy użyciu algorytmu PPO
do nauki gry w klasyczną grę Atari: "Atlantis".
Skrypt podzielony jest na dwie fazy:
1. Faza Treningu: Agent uczy się grać, analizując piksele z ekranu gry w wielu równoległych środowiskach.
   W tej fazie grafika jest wyłączona dla przyspieszenia obliczeń.
2. Faza Prezentacji: Po zakończeniu treningu uruchamiana jest instancja gry z włączoną grafiką
   gdzie wytrenowany model podejmuje decyzje w czasie rzeczywistym.

AUTOR:
Jakub Skarżyński, Sebastian Hellak

INSTRUKCJA UŻYCIA:
1. Upewnij się, że masz zainstalowane wymagane biblioteki. Wymaga to zainstalowania pakietów RL oraz ROM-ów Atari:
   pip install gymnasium[atari] stable-baselines3 shimmy ale-py
   pip install autorom[accept-rom-license]
2. Jeśli posiadasz kartę graficzną NVIDIA, upewnij się, że masz zainstalowany PyTorch z obsługą CUDA,
   aby trening przebiegał szybciej (parametr device="cuda").
3. Uruchom skrypt.
4. Aby przerwać działanie programu w fazie prezentacji, użyj w terminalu skrótu Ctrl+C.

REFERENCJE:
- Stable Baselines3 PPO: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
- Gymnasium Atari: https://gymnasium.farama.org/environments/atari/
- Gra Atlantis: https://gymnasium.farama.org/environments/atari/atlantis/
"""

import gymnasium as gym
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import time

ENV_NAME = "ALE/Atlantis-v5"
TRAINING_STEPS = 1_000_000
SHOW_PREVIEW = True

def main():
    """
    Główna funkcja sterująca procesem uczenia i ewaluacji.
    """
    print("=== FAZA 1: TRENING (BEZ GRAFIKI) ===")
    print(f"Rozpoczynam trening na {TRAINING_STEPS} kroków...")
    
    train_env = make_atari_env(ENV_NAME, n_envs=64, seed=42)
    
    train_env = VecFrameStack(train_env, n_stack=4)

    model = PPO(
        "CnnPolicy", 
        train_env, 
        verbose=1, 
        learning_rate=2.5e-4, 
        n_steps=128, 
        batch_size=256,
        ent_coef=0.01,
        device="cuda"
    )
    
    model.learn(total_timesteps=TRAINING_STEPS)
    print("Trening zakończony!")
    
    train_env.close()

    if SHOW_PREVIEW:
        print("\n=== FAZA 2: PREZENTACJA (Z GRAFIKĄ) ===")
        print("Ładowanie okna gry...")

        eval_env = make_atari_env(ENV_NAME, n_envs=1, seed=42, env_kwargs={"render_mode": "human"})
        
        eval_env = VecFrameStack(eval_env, n_stack=4)

        obs = eval_env.reset()
        
        try:
            while True:
                action, _ = model.predict(obs, deterministic=True)
                
                obs, rewards, dones, info = eval_env.step(action)
                
                if dones:
                    obs = eval_env.reset()
                    
        except KeyboardInterrupt:
            print("\nZatrzymano przez użytkownika.")
        finally:
            eval_env.close()

if __name__ == "__main__":
    main()
