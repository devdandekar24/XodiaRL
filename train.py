from env import PocketTank
import math
from stable_baselines3 import A2C
#dev_dandekar

# import os
# models_dir="models/A2C-m1"
# logdir="logs"
# if not os.path.exists(models_dir):
#     os.makedirs(models_dir)
# if not os.path.exists(logdir):
#     os.makedirs(logdir)

# CustomEnv inherits the PocketTank
class CustomEnv(PocketTank):
    # reward function written by me
    def _get_reward(self,diff,bullet_type):      
        #M1
        #healing bullet
        if bullet_type==5:
            reward=-1/diff
        #standard
        # high positive reward based on the closeness to the target
        elif bullet_type==6:
            reward=(20-diff)*25
        #boomerang
        # elif bullet_type==6:
        #     reward=2**diff
        #others
        else:
            reward=(1/(math.sqrt(diff)*10))-25
        return reward

env = CustomEnv() # instantiates the custom environment

bot = A2C(policy='MlpPolicy', env=env, verbose = 1) #initializes the A2C model with a multilayer perceptron policy ( use of neural networks), verbose=1 meaning it prints the info about the training process

# TIMESTEPS=25000

if __name__ == '__main__':
    # for i in range(1,20):
    bot.learn(total_timesteps=60000)
    bot.save("./trained_bots/bot")
