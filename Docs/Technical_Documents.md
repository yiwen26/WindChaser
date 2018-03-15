In this file we note the basic algorithms for our project both on forecasts and future decision making.

## LSTM
### Back Propagation
Back-propagation algorithm is used to minimize the loss function by finding the right set of weights. Actually it has become one of the cornerstones for the success of deep learning. The goal of the back-propagation algorithm is to optimize the loss over neurons by computing the partial derivatives of the loss function with respect to the weights and biases. Since the task is to minimize a function by finding an optimal parameter, using stochastic gradient descent is the best way to learn a weight and bias. Specifically, in LSTM with the introduction of input gate (i), forget gate (f), and output gate (o), at each timestep (with subscript t). Once updated the neuron weights, the new output and hidden states can be using feed-forward calculation as follows:

<div align=center><img src="https://github.com/yiwen26/WindChaser/blob/master/Docs/Back_propagation.gif"/></div>

Want more math?!
If you want to go deeper in the math, i.e. the algorithm used to compute gradients across multiple layers of neurons. Google it! or 
<a href="https://wiseodd.github.io/techblog/2016/08/12/lstm-backprop/">`click here!`</a>
It's useful if you want to re-implement gradient descent on your own, or understand how it is done. 


### Loss function
In our work, to predict the wind power at LSTM network, we will use mean squared error loss functions. The method of minimizing MSE is called Ordinary Least Squares (OSL). The basic principle of OSL is that the optimized fitting line should be a line which minimizes the sum of distance of each point to the corresponding forecasts result. The MSE loss function is calculated between LSTM's prediction and real values:


<div align=center><img src="https://github.com/yiwen26/WindChaser/blob/master/Docs/Loss%20Function%20eq.png"/></div>

where n is the batch size for stochastic gradient descent.


## Q-Learning
### Reward function
Suppose an agent is refered to one electricity user, who can make decision at each time slot to whether use electricity or not based on his 
judgement on the electricity price right now. He his an objective of minimizing the energy costs through a period of time. Suppose it has a 
state at time $t$ which is denoted by s_t, representing how much more electricity he need to use by the end of day. Every time he also needs
to make a decision $a_t \in {0,1}$, which represents his action at each time slot to use electricity of not (in future work, we will consider
change $a_t$ as a continuous variable to denote how much energy it uses at each time). There is also also price signal $p_t$ which
is a stochastic process (for simplicity, we can start from Markov process). Then based on price signal, the user needs to decide if he or she
wants to consume energy at current time slot.

We design the reward function $r(s_t, a_t)$ based on price signal and utility function $U(s_t, a_t)$ which denotes the user's satisfication
level on delaying previous set schedule. For instance, we user's utility is large if it's his/her dinner time and wants to use microware.
Yet the schedule for the washer has a flexible schedule which has a smaller utility. Then we have 

![Alt Text](https://github.com/yiwen26/WindChaser/blob/master/WindChaserModules/Reinforcement%20Learning/RL_equ2.gif)


### Q table and updates
We start by initializing the table to be uniform (all zeros), and then as we observe the rewards we obtain for various actions, 
we update the table accordingly. We will update the table using the Bellman Equation
![Alt Text](https://github.com/yiwen26/WindChaser/blob/master/WindChaserModules/Reinforcement%20Learning/RL_Equ1.gif)

And it can be proved by updating the Q table using Bellman Equation, we are able to fine the optimal Q table for agent to make decisions.

### Epsilon Greedy for action selection
The best lever is selected for a proportion 1-\epsilon  of the trials, and a level is selected at random (with uniform probability) for a proportion \epsilon. A typical parameter value might be  \epsilon =0.1}, and it helps the exploration of Q table.
