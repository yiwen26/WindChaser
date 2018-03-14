import numpy as np
import matplotlib.pyplot as plt
from numpy import  shape
import csv

#Initialize the Q table. Right now we assume everyday the maximized electricity usage is 12, 
#which can be set as user-defined parameters in the future implementation
Q=np.zeros([13,2])
gamma=0.8
alpha=0.8
np.random.seed(10)
T = 10000
lam = 2
rand = np.random.uniform(0,1,(T,1))

#Define the reward function: composed of price costs plus the user utility function to 
#denote the user's satisfication level of delayed electricity service
def reward(action, state, p_signal):
    if action==1:
        reward=-p_signal
        delayed_penalty=0
        reward-=lam*delayed_penalty
    else:
        reward=0
        delayed_penalty=1
        reward-=lam*delayed_penalty
    return reward

def reward2(action, state, p_signal):
    if action==1:
        reward=-p_signal+1.0/3
    elif action==2:
        reward=0
    else:
        reward=p_signal-1.0/3
    return reward  

#Every step check the states to decide if some of the actions are available
def available_actions(state):
    if state==0:
        action=[0]
    else:
        action=[0,1]

    return action

#Use epsilon greedy method to update the actions based on Q-value.
def sample_next_action(available_act, current_state, p):
    if(p<0.9):
        m=len(available_act)
        q_val=[]
        for i in range(m):
            q_val.append(Q[current_state, available_act[i]])
        max_val=np.argmax(q_val)
        next_action=available_act[max_val]

    else:
        next_action = int(np.random.choice(available_act, 1))
    return next_action

#Define the update functions for Q function.
def update(current_state, action, signal):
    reward_val=reward(action=action, state=current_state, p_signal=signal)
    max_value=-1e10
    for j in range(2):
        if Q[current_state, j]>max_value:
            max_value=Q[current_state, j]
    Q[current_state, action] += alpha*(reward_val + gamma * max_value-Q[current_state, action])
    return Q[current_state, action]

#Update the current electricity usage state based on current state and current chosen action
def update_state(current_state, action):
    if action==1:
        state=current_state-1
    else:
        state=current_state
    return state


#The main function
n_episodes=1e3 #Number of iterations.
current_state=12
print ("current state: %d"%current_state)
scores=[]
price_total=0.0
price_total_list=[]
price_total_naive=0.0
price_total_naive_list=[]
day=0

for episode in range(T):

    available_act = available_actions(current_state)
    price = np.random.choice(4, 1, p=[0.5000, 0.1000, 0.3000, 0.1000]) + 1 #Price signal for current step. We can either
    action = sample_next_action(available_act, current_state, rand[episode])
    current_state = update_state(current_state, action)
    #signal = np.random.choice(2,1, p=[0.6666, 0.3334])
    score = update(current_state, action, price)
    scores.append(score)

    
    #The first 5,000 episodes can be understood as training set. Then compare the result against fixed schedule costs.
    if episode >5000:
        hour=episode-(day-1)*24
        if hour==6:
            price_total_naive+=price*0.5
        elif hour==7:
            price_total_naive+=price*0.5
        elif hour==8:
            price_total_naive+=price*1
        elif hour==9:
            price_total_naive+=price
        elif hour==10:
            price_total_naive+=price*2
        elif hour==11:
            price_total_naive+=price*2
        elif hour==12:
            price_total_naive+=price
        elif hour==13:
            price_total_naive+=price
        elif hour==14:
            price_total_naive+=price
        elif hour==15:
            price_total_naive+=price*2

    if action==1:
        #print("here: buy electricity")
        #print("Episode", episode)
        if episode>5000:
            price_total+=price
    #print ('Score:', str(score))
    if episode%24==0:
        print("Current state:", current_state)
        if episode >5000:
            price_total+=current_state*price
            price_total_list.append(np.copy(price_total))
            print("Naive price", np.copy(price_total_naive))
            print("Learned price", price_total)
            price_total_naive_list.append(np.copy(price_total_naive))
        current_state=12
        day+=1


print ("price total now:", price_total)
print ("price total naive: ", price_total_naive)

print("Final Q table", Q)

#Show the final accumulated price comparison plot
price_total_naive_list=np.array(price_total_naive_list, dtype=float)
price_total_list=np.array(price_total_list, dtype=float)
plt.plot(price_total_naive_list, 'r', linewidth=2.0)
plt.plot(price_total_list, 'b', linewidth=2.0)
plt.show()

#Compare and save the result
with open('naive_price.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(price_total_naive_list)


with open('learned_price.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(price_total_list)
