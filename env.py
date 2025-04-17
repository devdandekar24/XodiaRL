from gymnasium import Env, spaces
import random
import math
import numpy as np
import pygame
import time

pygame.init()
class PocketTank(Env):
    def __init__(self):
        self.tank_1_positions_range = (50,150)
        self.tank_2_positions_range = (650,750)

        self.x1 = random.randint(*self.tank_1_positions_range)
        self.x2 = random.randint(*self.tank_2_positions_range)
        self.g = 12

        self.observation_space = spaces.MultiDiscrete([801,801,7]) #range1, range2, bullet type
        self.action_space = spaces.MultiDiscrete([100,90,3])    #velocity, angle, 

        self.action_cnt = 20  # each player get total of 20 chances to do action
        self.remaining_actions = [self.action_cnt, self.action_cnt]

        self.v_wind = +8    # wind speed is of 8
        self.bullet_type = random.randint(0,6)  # random bullets are fired from 7 types

        self.w, self.h = 800, 500   # screen size
        self._init_display()
        self.episodes = 0   #episode counter for RL, one episode is full play of game
        self.reward = 0     #reward for current step
        self.tank_width = 56    # sets the tank width to 56
        self.bullet_dist= 0     # tracks how far a bullet is fired 

    def _get_boomerang_range(self, v, theta, g, air_factor = 7):
        theta = theta*math.pi/180 # degree to radians conversion
        if v*math.cos(theta)>0:         # shot right
            dir = 1
        elif v*math.cos(theta)<0:       # shot left
            dir = -1
        else:                           # shot straight up
            dir = 0
        # formula for Range
        range = (v**2)*math.sin(2*theta)/g - dir*0.5*air_factor*(v*math.cos(theta))**2/g**2
        return range
    
    def _get_range(self,action,tank,bullet_type):
        (v,angle,move) = action         # action is tuple with three values
        if(bullet_type==6):             # if bullet is of boomerang type
            g = self.g
            range = self._get_boomerang_range(v,angle,g)
            return range
        if(bullet_type==5):             # if bullet is heavy bullet
            v = min(50,v)
        rad_angle = math.radians(angle) # angle to radians conversion
        vx = v * math.cos(rad_angle)    # x comp of velocity
        vy = v * math.sin(rad_angle)    # y comp of velocity

        # add/sub wind to vx based on which tank is shooting
        if(tank==0):
            vx = vx + self.v_wind
        else : 
            vx = vx - self.v_wind
        # range formula
        range = ((2 * vy)/self.g) * vx
        return range

    def _get_bullet_position(self,action,tank,bullet_type): # calculate final hori pos of bullet
        range = self._get_range(action,tank,bullet_type)  # calculate how far a bullet travel horizontally
        x_bullet = 0     # init bullet position
        if(tank==0):    # tank at left
            x_bullet =  self.x1 + range
        else :          # tank at right
            x_bullet =  self.x2 - range
        return x_bullet

    def _get_diff(self,x_bullet,tank):      # calc how far is bullet from opponents tank
        diff = 0    # diff be zero
        if(tank==0):    # for tank at left
            diff = abs(self.x2-x_bullet)    # x2 - xpos of bullet
        else: 
            diff = abs(self.x1-x_bullet)    # x1- xpos of bullet
        return diff
# ******************************************************************************************** #
    def _get_reward(self,diff, bullet_type): # custom reward logic
        # reward = ....
        return 0

    def _check_for_end(self):    # ends the game if one or both player completes all the moves
        # cond1 : both players are out of moves
        # cond2 : player 0 used all moves while player 1 didnt 
        if(self.remaining_actions == [0,0] or self.remaining_actions==[0,self.action_cnt]):
            return True
        else : 
            return False

    def _get_state(self):       # return state, pos of tank 1 and 2 and bullet type in use
        state = np.array([self.x1,self.x2,self.bullet_type])
        return state

    def _make_move(self,move,tank): # updates the tanks horizontal pos based on the move dir
        # prevents the overlap and crossing at middle
        if(tank==0):
            # for tank on left
            # move by 25 units
            # left min=0 and right max=300
            if(move==0):
                self.x1 = min(self.x1 + 25, 300)
            elif (move == 1):
                self.x1 = max(self.x1-25,0)
        else : 
            # for tank on right
            # left min=500 and right max=798
            if(move==0):
                self.x2 = max(500,self.x2-25)
            elif(move == 1) : 
                self.x2 = min(798,self.x2+25)

    def step(self, action, tank=0): # represents one step in the game where the agent takes the action and the env updates accordingly
        (v,angle,move) = action
        self._make_move(move,tank) # move the tank
        bullet_type = self.bullet_type 
        x_bullet = self._get_bullet_position(action,tank,bullet_type) # final hori pos of bullet fired
        diff = self._get_diff(x_bullet,tank) # diff in bullets and target tanks pos
        self.bullet_dist = diff     # update bullet dis
        reward = self._get_reward(diff,bullet_type)  #calculate the reward
        
        self.remaining_actions[tank]-=1     # update the remaining actions count
        done = self._check_for_end()        # check if it is end
        self.bullet_type = random.randint(0,6)  # randomize the bullet
        state = self._get_state()  #calc the new state
        # adding the information
        info = {"reward":reward}
        if(diff < self.tank_width/2):
            info["hit"]=True
        else : 
            info["hit"]=False
        truncated = False # to signal if the game was truncated

        self.timesteps += 1 # inc no of timesteps
        self.reward += reward #accumulate the total reward
        # checking when to render the game and when to save specific moments
        # it shows only first 10 actions in the first 100 episodes
        if self.episodes % 100 == 0 and self.timesteps < 10:
            self.bullet_buffer = self._get_bullet_pts(action)
            self.x_bullet = x_bullet
            self.turn = tank
            self.action = action
            self.render()
        
        return (state,reward,done,truncated,info)

    def reset(self,seed=None, values = None):  #call at begining of each epoch to reset the game
        self.x1 = random.randint(*self.tank_1_positions_range)
        self.x2 = random.randint(*self.tank_2_positions_range)
        self.remaining_actions = [self.action_cnt, self.action_cnt]
        state = self._get_state()
        info = {}

        self.episodes += 1
        self.timesteps = 0
        self.reward = 0
        self.bullet_dist = 0
        return (state,info)
    
    # setting up game window using pygame
    def _init_display(self):
        self.display = pygame.display.set_mode((self.w, self.h)) # setting window
        pygame.display.set_caption('TANKS')     # title of window
        self.clock = pygame.time.Clock()        # clock to control how fast the game updates
        self.font = pygame.font.SysFont('arial', 25)    # size and font style of any text
        self.FRAME_RATE = 60   # 60 fps

    def _get_bullet_pts(self, action):          # calculates and returns a list of x,y pos that represent the trajectory of a bullet when its fired
        pointsBuffer = [] 
        (v,angle,move) = action
        angle = math.radians(angle)
        # splitting velocity to horizontal and vertical
        vx = v * math.cos(angle)
        vy = v * math.sin(angle)
        vx = vx + self.v_wind           # adding wind effect to speed 
        # t is simulated time, 30 time steps scaled by 0.5 to get more pts
        for t in range(30):
            t = 0.5*t
            # calculating the new x and y positions
            x_new = (self.x1)+vx*t
            y_new = 50+vy*t-0.5*self.g*t**2
            # if bullet hits the ground, stop and return the path so far
            if y_new < 0:
                return pointsBuffer
            # add x and y to the path of bullet
            pointsBuffer.append((x_new, y_new))
        return pointsBuffer

    # drawing the game screen using pygame
    def render(self):
        self.display.fill((30, 30, 80))     # fill the bg with blue coolor
        pygame.draw.rect(self.display, (50, 0, 0), (self.x1-28, self.h-100, 56, 50)) # drawing the red tank
        pygame.draw.rect(self.display, (0, 0, 50), (self.x2-28, self.h-100, 56, 50)) # drawing the blue tank
        # drawing the borders around the tank to highlight them
        pygame.draw.rect(self.display, (140, 0, 0), (self.x1-28, self.h-100, 56, 50), 2)
        pygame.draw.rect(self.display, (0, 0, 140), (self.x2-28, self.h-100, 56, 50), 2)
        # green strip at the bottom of the screen
        pygame.draw.rect(self.display, (20, 80, 0), pygame.Rect(0, self.h-50, self.w, 50))
        self.__draw_gui() # helper function to draw text/ info
        # drawing bullets 
        pygame.draw.circle(self.display, (255, 255, 255), (self.x_bullet, 500-50), 5)
        pygame.draw.circle(self.display, (255, 255, 255), (self.x_bullet, 500-50), 8, 2)
        # drawing the bullets trajectory
        for i, point in enumerate(self.bullet_buffer):
            point = (point[0], 500-point[1])   # flipping the y axis
            pygame.draw.circle(self.display, (255, 150, 150), (point[0], point[1]), 3)  # draw the dot


        pygame.display.flip() # update the screen and slow down
        self.clock.tick(self.FRAME_RATE)    #ensure that game doesnot run faster than the frame_rate
        time.sleep(1)   # pauses 1 sec so that you can see bullet being drawn
        self.__handle_events() # handles keyboard and mouse events

    def __handle_events(self):      # handling the closing of the game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    quit()

    def __draw_gui(self):    # drawing all the onscreen game info
        bullet_names = ['Standard Shell', 'Triple Threat', 'Long Shot', 'Blast Radius', 'Healing Halo', 'Heavy Impact', 'Boomerang Blast']
        bullet_colors = [(176, 196, 222), (147, 112, 219), (0, 139, 139), (255, 140, 0), (0, 250, 154), (178, 34, 34), (148, 0, 211)]
        pygame.draw.rect(self.display, (100, 130, 160), (5, self.h-5, self.w - 10, 200), 2) # light blue rectangle at the bottom part of the screen
        # episodes and timestamps passed
        text = self.font.render(f'Episode: {self.episodes} ({self.timesteps}/20)', True, (255, 255, 255))
        self.display.blit(text, [300, 10])
        # total reward earned and how far did the bullet landed from the tank
        text = self.font.render(f'Reward: {str(self.reward)[:5]} Diff: {str(self.bullet_dist)}', True, (255, 255, 255))
        self.display.blit(text, [300, 60])
        #  display current bullet type
        text = self.font.render(bullet_names[self.bullet_type], True, bullet_colors[self.bullet_type])
        self.display.blit(text, [20, 50])
        # power of current shot
        text = self.font.render(f'Power: {self.action[0]}', True, (255, 0, 0))
        self.display.blit(text, [20, 80])
        # angle of current shot
        text = self.font.render(f'Angle: {self.action[1]}', True, (180, 255, 250))
        self.display.blit(text, [20, 110])