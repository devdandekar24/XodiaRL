
# playing game with single bot
def single_bot_game(env,bot,episodes = 10):
    hit_cnt = 0  #count of shots hit the enemy
    for i in range (episodes): # looping through each episode / game
        done = False # not done yet
        (state,info) = env.reset() # reset the env for new game
        score = 0  # track the total reward earned
        while(done == False):
            (action,value) = bot.predict(state) # predict action based on current state
            (state,reward,done,truncated,info) = env.step(action) # take action in the environment
            score += reward # update the score
            if(info['hit']==True): # if hit then update hit_cnt
                hit_cnt+=1
        print(f"Episode {i+1} score: {score}") # print the score and episode
    print("Hit percentage : ", hit_cnt/(episodes * env.action_cnt))

def get_updated_state(state): # to flip state so that the tank2 can see game with its own perspective
    updated_state = [800-x for x in state[:-1]]
    updated_state.reverse() # reverse order
    updated_state.append(state[-1])   # add back original bullet type
    return updated_state
    # example: [100,700,3] => [800-100,800-700]->[700,100] => reversed [100,700] => add bullet [100,700,3]


# 2 player game
def compete_bots(env, bot_1,bot_2, episodes=10):
    hit_cnt_1 = 0
    hit_cnt_2 = 0
    for i in range(episodes):
        done = False
        (state,info) = env.reset()
        score_1 = 0
        score_2 = 0 
        while(done == False): # while game is not over
            (action,value) = bot_1.predict(state) # bot1 takes an action
            (state,reward,done,truncated,info) = env.step(action,0) # then env is updated based on action
            score_1 += reward # update the scroe
            if(info['hit']==True): # if hit then increase the hit count
                hit_cnt_1+=1
            updated_state = get_updated_state(state) # now update the state for bot2
            (action,value) = bot_2.predict(updated_state) # again pick an action based on the model 
            (state,reward,done,truncated,info) = env.step(action,1) # update env according to action
            score_2 += reward # update the score
            if(info['hit']==True): # if hit, increase the cnt
                hit_cnt_2+=1
        # print scores for each episode 
        print(f"Episode {i+1} score -> bot 1 score : {score_1} | bot 2 score : {score_2} ")
    print("Tank 1 hit percentage : ", hit_cnt_1/(episodes*env.action_cnt))
    print("Tank 2 hit Percentage : ", hit_cnt_2/(episodes*env.action_cnt))

