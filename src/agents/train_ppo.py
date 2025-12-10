"""
Train a PPO agent on the PortfolioEnv using the training data.

Steps
- load prices and split train / test
- build technical features on the train period
- create the gym environment
- train PPO
- save the trained model to disk
"""

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.config import MODELS_DIR, PPO_MODEL_PATH
from src.data.load_data import load_prices
from src.data.preprocess import split_train_test
from src.features.technical_indicators import compute_technical_features
from src.env.portfolio_env import PortfolioEnv


def make_train_env(prices_train, features_train):
    
    #Return a function that creates a PortfolioEnv instance.

    #This is used by DummyVecEnv to build the vectorized environment.
    

    def _init():
        # Align prices with the feature index
        prices_aligned = prices_train.loc[features_train.index]
        env = PortfolioEnv(
            prices=prices_aligned,
            features=features_train,
            initial_capital=1.0,
            transaction_cost=0.001,  # 0.1% transaction cost per unit turnover
        )
        return env

    return _init


def train_ppo(
    total_timesteps: int = 100_000,
    seed: int = 42,
) -> None:
    #Train a PPO agent on the training period and save the model.

    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load prices and split into train / test
    prices = load_prices()
    prices_train, _ = split_train_test(prices)

    # Compute technical features on the train set
    features_train = compute_technical_features(prices_train)

    # Build vectorized environment
    train_env_fn = make_train_env(prices_train, features_train)
    env = DummyVecEnv([train_env_fn])

    # Create PPO model (quiet mode: verbose=0)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=0,  # no training table printed
        seed=seed,
    )

    # Train the agent
    model.learn(total_timesteps=total_timesteps)

    # Save model to disk
    model.save(str(PPO_MODEL_PATH))
    print("PPO training finished")
    print("Model saved to:", f"{PPO_MODEL_PATH}.zip")


def main():
    # Default training run
    train_ppo(total_timesteps=100_000, seed=42)


if __name__ == "__main__":
    main()