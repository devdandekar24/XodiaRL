from stable_baselines3 import A2C   # using A2C RL algorithms
from train import CustomEnv  # importing game env class defined in train.py
from utils import * # importing helper functions from utils.py


env = CustomEnv() # creating a new env
# paths to the trained bots
bot_1_path = "./trained_bots/bot.zip"   
bot_2_path = "./trained_bots/bot.zip"
# loading and attaching the bots into the memory
bot_1 = A2C.load(bot_1_path, env=env)
bot_2 = A2C.load(bot_2_path, env=env)
# playing the game for 5 episodes/ matches
episodes = 5
single_bot_game(env,bot_1,episodes) # single player game
compete_bots(env,bot_1,bot_2,episodes) # bot 1 vs bot 2 for 5 episodes