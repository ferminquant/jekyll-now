My summary of the book "Reinforcement Learning - An Introduction by Richard S. Sutton and Andrew G. Barto" found [here](http://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html).

I write this as my notes as I go through the book from cover to cover. The purpose is to have them as personal reference notes, not to re-explain what is already in the book. However, they might be useful for someone else.

I found part I and II better explained and more useful. Part III seems to be more about what things can be advanced through more research, and it felt like the authors didn't have such a great grasp of the concepts explained like in Part I and II, or at least they didn't bother explaining with the same level of quality. I think like this because one of my favorite quotes is from Albert Einstein which states "If you can't explain it simply, you don't understand it well enough". Parts I and II were explained simply, part III not so much.

Note: I wrote gamma instead of lambda in quite a few cases. When I realized, I just started using lambda but left the previous incorrect uses of gamma unchanged.

<!-- MarkdownTOC autolink="true" bracket="round" depth="0" style="unordered" indent="  " autoanchor="false" -->

- [I. The Problem](#i-the-problem)
  - [Chapter 1 Introduction](#chapter-1-introduction)
    - [1.1 Reinforcement Learning](#11-reinforcement-learning)
    - [1.2 Examples](#12-examples)
    - [1.3 Elements of Reinforcement Learning](#13-elements-of-reinforcement-learning)
    - [1.4 An Extended Example: Tic-Tac-Toe](#14-an-extended-example-tic-tac-toe)
    - [1.5 Summary](#15-summary)
    - [1.6 History of Reinforcement Learning](#16-history-of-reinforcement-learning)
    - [1.7 Bibliographical Remarks](#17-bibliographical-remarks)
  - [Chapter 2 Evaluative Feedback](#chapter-2-evaluative-feedback)
    - [2.1 An n-Armed Bandit Problem](#21-an-n-armed-bandit-problem)
    - [2.2 Action-Value Methods](#22-action-value-methods)
    - [2.3 Softmax Action Selection](#23-softmax-action-selection)
    - [2.4 Evaluation Versus Instruction](#24-evaluation-versus-instruction)
    - [2.5 Incremental Implementation](#25-incremental-implementation)
    - [2.6 Tracking a Nonstationary Problem](#26-tracking-a-nonstationary-problem)
    - [2.7 Optimistic Initial Values](#27-optimistic-initial-values)
    - [2.8 Reinforcement Comparison](#28-reinforcement-comparison)
    - [2.9 Pursuit Methods](#29-pursuit-methods)
    - [2.10 Associative Search](#210-associative-search)
    - [2.11 Conclusions](#211-conclusions)
    - [2.12 Bibliographical and Historical Remarks](#212-bibliographical-and-historical-remarks)
  - [Chapter 3 The Reinforcement Learning Problem](#chapter-3-the-reinforcement-learning-problem)
    - [3.1 The Agent-Environment Interface](#31-the-agent-environment-interface)
    - [3.2 Goals and Rewards](#32-goals-and-rewards)
    - [3.3 Returns](#33-returns)
    - [3.4 Unified Notation for Episodic and Continuing Tasks](#34-unified-notation-for-episodic-and-continuing-tasks)
    - [3.5 The Markov Property](#35-the-markov-property)
    - [3.6 Markov Decision Processes](#36-markov-decision-processes)
    - [3.7 Value Functions](#37-value-functions)
    - [3.8 Optimal Value Functions](#38-optimal-value-functions)
    - [3.9 Optimality and Approximation](#39-optimality-and-approximation)
    - [3.10 Summary](#310-summary)
    - [3.11 Bibliographical and Historical Remarks](#311-bibliographical-and-historical-remarks)
- [II. Elementary Solution Methods](#ii-elementary-solution-methods)
  - [Chapter 4 Dynamic Programming](#chapter-4-dynamic-programming)
    - [4.1 Policy Evaluation](#41-policy-evaluation)
    - [4.2 Policy Improvement](#42-policy-improvement)
    - [4.3 Policy Iteration](#43-policy-iteration)
    - [4.4 Value Iteration](#44-value-iteration)
    - [4.5 Asynchronous Dynamic Programming](#45-asynchronous-dynamic-programming)
    - [4.6 Generalized Policy Iteration](#46-generalized-policy-iteration)
    - [4.7 Efficiency of Dynamic Programming](#47-efficiency-of-dynamic-programming)
    - [4.8 Summary](#48-summary)
    - [4.9 Bibliographical and Historical Remarks](#49-bibliographical-and-historical-remarks)
  - [Chapter 5 Monte Carlo Methods](#chapter-5-monte-carlo-methods)
    - [5.1 Monte Carlo Policy Evaluation](#51-monte-carlo-policy-evaluation)
    - [5.2 Monte Carlo Estimation of Action Values](#52-monte-carlo-estimation-of-action-values)
    - [5.3 Monte Carlo Control](#53-monte-carlo-control)
    - [5.4 On-Policy Monte Carlo Control](#54-on-policy-monte-carlo-control)
    - [5.5 Evaluating One Policy While Following Another](#55-evaluating-one-policy-while-following-another)
    - [5.6 Off-Policy Monte Carlo Control](#56-off-policy-monte-carlo-control)
    - [5.7 Incremental Implementation](#57-incremental-implementation)
    - [5.8 Summary](#58-summary)
    - [5.9 Bibliographical and Historical Remarks](#59-bibliographical-and-historical-remarks)
  - [Chapter 6 Temporal-Difference Learning](#chapter-6-temporal-difference-learning)
    - [6.1 TD Prediction](#61-td-prediction)
    - [6.2 Advantages of TD Prediction Methods](#62-advantages-of-td-prediction-methods)
    - [6.3 Optimality of TD\(0\)](#63-optimality-of-td0)
    - [6.4 Sarsa: On-Policy TD Control](#64-sarsa-on-policy-td-control)
    - [6.5 Q-Learning: Off-Policy TD Control](#65-q-learning-off-policy-td-control)
    - [6.6 Actor-Critic Methods](#66-actor-critic-methods)
    - [6.7 R-Learning for Undiscounted Continuing Tasks](#67-r-learning-for-undiscounted-continuing-tasks)
    - [6.8 Games, Afterstates, and Other Special Cases](#68-games-afterstates-and-other-special-cases)
    - [6.9 Summary](#69-summary)
    - [6.10 Bibliographical and Historical Remarks](#610-bibliographical-and-historical-remarks)
- [III. A Unified View](#iii-a-unified-view)
  - [Chapter 7 Eligibility Traces](#chapter-7-eligibility-traces)
    - [7.1 n-Step TD Prediction](#71-n-step-td-prediction)
    - [7.2 The Forward View of TD\(gamma\)](#72-the-forward-view-of-tdgamma)
    - [7.3 The Backward View of TD\(gamma\)](#73-the-backward-view-of-tdgamma)
    - [7.4 Equivalence of Forward and Backward Views](#74-equivalence-of-forward-and-backward-views)
    - [7.5 Sarsa\(gamma\)](#75-sarsagamma)
    - [7.6 Q\(gamma\)](#76-qgamma)
    - [7.7 Eligibility Traces for Actor-Critic Methods](#77-eligibility-traces-for-actor-critic-methods)
    - [7.8 Replacing Traces](#78-replacing-traces)
    - [7.9 Implementation Issues](#79-implementation-issues)
    - [7.10 Variable Lambda](#710-variable-lambda)
    - [7.11 Conclusions](#711-conclusions)
    - [7.12 Bibliographical and Historical Remarks](#712-bibliographical-and-historical-remarks)
  - [Chapter 8 Generalization and Function Approximation](#chapter-8-generalization-and-function-approximation)
    - [8.1 Value Prediction with Function Approximation](#81-value-prediction-with-function-approximation)
    - [8.2 Gradient-Descent Methods](#82-gradient-descent-methods)
    - [8.3 Linear Methods](#83-linear-methods)
      - [8.3.1 Coarse Coding](#831-coarse-coding)
      - [8.3.2 Tile Coding](#832-tile-coding)
      - [8.3.3 Radial Basis Functions](#833-radial-basis-functions)
      - [8.3.4 Kanerva Coding](#834-kanerva-coding)
    - [8.4 Control with Function Approximation](#84-control-with-function-approximation)
    - [8.5 Off-Policy Bootstrapping](#85-off-policy-bootstrapping)
    - [8.6 Should We Bootstrap?](#86-should-we-bootstrap)
    - [8.7 Summary](#87-summary)
    - [8.8 Bibliographical and Historical Remarks](#88-bibliographical-and-historical-remarks)
  - [Chapter 9 Planning and Learning](#chapter-9-planning-and-learning)
    - [9.1 Models and Planning](#91-models-and-planning)
    - [9.2 Integrating Planning, Acting, and Learning](#92-integrating-planning-acting-and-learning)
    - [9.3 When the Model is Wrong](#93-when-the-model-is-wrong)
    - [9.4 Prioritized Sweeping](#94-prioritized-sweeping)
    - [9.5 Full vs. Sample Backups](#95-full-vs-sample-backups)
    - [9.6 Trajectory Sampling](#96-trajectory-sampling)
    - [9.7 Heuristic Search](#97-heuristic-search)
    - [9.8 Summary](#98-summary)
    - [9.9 Bibliographical and Historical Remarks](#99-bibliographical-and-historical-remarks)
  - [Chapter 10 Dimensions of Reinforcement Learning](#chapter-10-dimensions-of-reinforcement-learning)
    - [10.1 The Unified View](#101-the-unified-view)
    - [10.2 Other Frontier Dimensions](#102-other-frontier-dimensions)
  - [Chapter 11 Case Studies](#chapter-11-case-studies)
    - [11.1 TD-Gammon](#111-td-gammon)
    - [11.2 Samuel's Checkers Player](#112-samuels-checkers-player)
    - [11.3 The Acrobot](#113-the-acrobot)
    - [11.4 Elevator Dispatching](#114-elevator-dispatching)
    - [11.5 Dynamic Channel Allocation](#115-dynamic-channel-allocation)
    - [11.6 Job-Shop Scheduling](#116-job-shop-scheduling)
  - [Bibliography](#bibliography)

<!-- /MarkdownTOC -->

# I. The Problem

## Chapter 1 Introduction
We learn by interacting with the environment and receiving its feedback. There is no teacher, just feedback. 
Reinforcement learning is a computational approach to this type of learning.

### 1.1 Reinforcement Learning
Learn what to do in order to maximize a numerical reward signal.
It is different from supervised learning since it does not learn from examples.
One of the challenges is the trade-off of exploration and exploitation.
It addresses the whole problem of interaction in an unknown environment, not just a subproblem.

### 1.2 Examples
Playing chess, you get feedback by winning or losing.
Petroleum refinery operation, you monitor the output by changing the parameters.
A calf learning to walk right after being born.
A robot deciding to continue working or going back to the charge station.
Preparing breakfast.
In the examples the agent uses experience to better optimize future results.

### 1.3 Elements of Reinforcement Learning
Agent, environment, policy, reward function, value function, model (optional).
- Policy: the agent's way of behaving at a given time
- Reward function: the goal is to maximize this. The short term immediate gain of an action.
- Value function: what is good in the long run, based on reward.
- Model: mimics the environment, used for planning.

### 1.4 An Extended Example: Tic-Tac-Toe
Uses tic-tac-toe as a reinforcement learning example. Is has different states, and an outcome, while playing games, it updates the value estimate of each state based on the eventual result of winning or not.
Different from genetic algorithms (evolutionary methods), they consider a policy then play many games, ignoring what happens during the game. Value function evaluates every state's possible actions.

### 1.5 Summary
Reinforcement learning is a computational approach for an agent in an unknown environment to learn to optimize a numerical goal by interacting with the environment and getting feedback from it.

### 1.6 History of Reinforcement Learning
Three threads:
- Trial and error
- Problem of optimal control and its solution using value functions and dynamic programming
- Temporal difference methods
Because of the trail-and-error concept it was confused with supervised learning before being properly separated by the authors. All three threads were considered different before this.

### 1.7 Bibliographical Remarks
Some reference material was presented here.

## Chapter 2 Evaluative Feedback
You evaluate the actions taken, instead of instructing the algorithm which action to take. This is the difference with supervised learning.

### 2.1 An n-Armed Bandit Problem
This is the one-armed bandit but with n arms. So you have n options, and every time you take one you evaluate is estimated value versus its real value.
Talks about the conflict of exploration and exploitation:
- Exploitation: take the option with the highest value estimate.
- Exploration: take an option which is not the highest value estimate, maybe you will find it is actually better.

### 2.2 Action-Value Methods
Q*(a) - the true value of an action, which is the mean reward received when selected.
Qt(a) - the estimated value at the t-th action
Choose most of the time the greedy option (the action with the highest value)
Choose a non optimal action with a small probability (epsilon).
These methods are called epsilon-greedy.
Three cases were compared;
1 - greedy or epsilon = 0
2 - epsilon = 0.1
3 - epsilon = 0.01
Option 1 from a good option really fast and stayed with it. 
Option 2 took a little longer to find a good reward but over time was better. 
Option 3 took even longer, but when found would exploit it more often than option 2.
The epsilon parameter is the measure of exploration vs exploitation. You could change epsilon over time. It is important to consider when and if the values of actions changes over time, so as to always keep an effective epsilon for exploration and updates those values accordingly.

### 2.3 Softmax Action Selection
An epsilon-greedy chooses randomly between all non-optimal actions. Meaning it choose the worst action just as much as the second best. This is bad when the worst action is considerably worse. So a softmax action ranks the actions based on how good they are, and chooses the best ones more frequently than the worst ones when exploring.
The most common softmax method uses a Gibbs, or Boltzmann, distribution.
According to the book when it was written, there is no study that shows if a softmax selection is better or not than an epsilon-greedy selection.

### 2.4 Evaluation Versus Instruction
Evaluation learning - is just what previous examples have done, they make an action and evaluate how good it was. To know if it was correct or not it needs to evaluate all actions a lot of times. 
Instruction learning - what supervised learning does, you need not search the actions, you are told which is the best action after you take on. There is no searching of actions. You might search the parameter space, but not the action space.
Supervised learning doesn't control its environment because it follows (not influences) the information it receives. Instead of trying to make its environment behave in a certain way, it tries to make itself behave as instructed by its environment.

For binay problems (two actions) if deterministic (not random), supervised learning works great. If stochastic (random), then not. Because if both are > 0.5, one 0.8 and the other 0.9, then it might get incorrectly stuck in the 0.8 one, believing the other to be 0.2, but its not because it is stochastic. 
As a conclusion, this shows that we need something different than just supervised learning.

### 2.5 Incremental Implementation
Instead of storing all rewards for an action and re-calculate its value as an average, just do an incremental update. The general form is:
NewEstimate = OldEstimate + StepSize[Target - OldEstimate]
StepSize changes, it is 1/k, where k is the (k+1)st reward.

### 2.6 Tracking a Nonstationary Problem
The previous scenarios are for stationary problems, the bandit does not change over time. 
For non-stationary problems, we need to give more weight to recent rewards than to previous ones. Thus, we use a constant step-size called alpha (0 < alpha <= 1).
Sometimes alpha is not constant, but seldom used in real-life problems.

### 2.7 Optimistic Initial Values
We have needed to give initial value estimates a value, this is bias. When all options are tried at least once, bias disappears. However, with a constant alpha, it is always there, but slowly decreasing with time. In practice, it is not a problem.
Could be used to encourage exploration, setting +5 as default but with mean reward of 0 and variance 1. This is called optimistic initial value. At the start it performs worse, and with time it performs better, since it explored more it found the best action faster. It is an effective trick on stationary problems. Not useful for non-stationary, since the encouraging of exploration is temporary.

### 2.8 Reinforcement Comparison
Reference reward: the reference to know if a received reward is good or bad. Could be the average of previous rewards. The mothods who use this are called reinforcement comparison methods, sometimes more effective than action-value methods, and precursors to actor-critic methods (presented later, they solve the full reinforcement learning problem).
Reinforcement comparison methods do not maintain action-value estimates, only an overall reward level. Like softmax, it kind of ranks the actions every time it receives a new reward from an action.

### 2.9 Pursuit Methods
Pursuit methods maintain both action-value estimates and action preferences. 
In the tests of the book, it performed better that all previous methods covered thus far. But it is not always the case, each method has its own uses and advantages.
The example in the book is for stationary environments.

### 2.10 Associative Search
Non-associative tasks, no need to associate actions with situations.
Example, multiple n-armed bandit problems, uniquely identified by color, so you have if red then 1, if blue then 2, etc. 
These are called associative search tasks. They are intermmediate between n-armed bandit and full reinforcement learning problems:
- Full problem: it learns a policy
- n-armed problem: each action affects only the immediate reward
It becomes a full reinforcement learning problem if actions are allowed to affect the next situation as well as the reward.

### 2.11 Conclusions
Methods to balance exploration and exploitation:
- epsilon greedy
- softmax
- pursuit
Interval estimation methods: for each action estimate a confidence interval of its value. At the moment, it is not feasible in practice.

### 2.12 Bibliographical and Historical Remarks
Reference material is presented here.

## Chapter 3 The Reinforcement Learning Problem
The problem the book tries to solve will be introduced, as well as its trade-offs and challenges.

### 3.1 The Agent-Environment Interface
Agent: the learner and decision-maker.
Environment: the thing the agent interacts with.
They interact continually. Environment is affected by actions and responds with a reward and a new state.
Each state is mapped to probabilities to select an action, this is called the policy.
Sensory mechanisms are considered part of the environment, not the agent. If it cannot be changed by the agent, it is part of the environment.
The agent-environment boundary represents the limit of the agent's absolute control, not of its knowledge.

### 3.2 Goals and Rewards
The goal of the agent is to maximize the cumulative reward over time.
The reward is to tell the agent what to acheive, not how. Reward in chess is for winning, not for capturing opponent pieces.

### 3.3 Returns
We want to maximize the expected return, which in the simplest case is the sum of the rewards fromt the first state to the terminal state. Tasks with a terminal state are called episodic tasks.
If no terminal task, they are called continuing tasks.
They use discounting, for expected return, called discounted return. Using gamma where 0<=gamma<=1 as the discount rate. It determines the present values of future rewards. The closer to 1 the more farsighted, the closer to 0 the more nearsighted.

### 3.4 Unified Notation for Episodic and Continuing Tasks
It just clarifies a notation that they will use in the rest of the book to reference both types of tasks (episodic and continuing).

### 3.5 The Markov Property
Markov property, a property of the states. State is whatever information is available to the agent.
The state is not only the immediate sensation, For example, you can move your eyes to see a whole image, but only one part at a time, the state is the whole image. The state does not necessarily contain everything that is useful to know. Agent is not penalized for not knowing something, only for forgetting something.
A Markov state is a state that has all relevant information about past states. Examples, checkers board state, most is lost but everything necessary for the future is still there. Also, velocity and position of a cannonball.
The closest the state is to being Markov, the more useful it is, and the better performance will be acheived by the reinforcement learning systems. The book assumes all states to be Markov.
Most of the times, we don't have a complete Markov state, but we try to have it as close as possible to Markov, and it usually works pretty good. So not having a Markov state is not a severe problem.

### 3.6 Markov Decision Processes
MDP, Markov Decision Process, a reinforcement learning task that satisfies the Markov property. If state and actions are finite it is a finit MDP.
Given a state and action, each probability for next state.

### 3.7 Value Functions
Value functions are used to estimate the expected return of any action in a state. 
Monte Carlo methods are when you keep averages for each action taken in a state.
Various examples are given, including a rough version of GridWorld.

### 3.8 Optimal Value Functions
Optimal policy is the one which gives the most reward in the long run.
You can use Bellman optimality equation to find the optimal policy. It is impractical as it performs an exhaustive search and assumes the following which rarely happen in the real world scenarios:
- we accurately know the dynamics of the environment
- we have enough computational resources to complete the computation of the solution
- the Markov property

### 3.9 Optimality and Approximation
Finding the optimal policy is the best you can do, but almost never possible. Computational, memory and time constraints limit doing this, hence, we need to approximate.
Reinforcement learning approximates by learning for states that happen the most, and learning less for those that almost do not happen. It is an ok trade-off and an important diffentiating characteristic in comparison to other approaches to approximate the solving of MDPs.

### 3.10 Summary
Agent, environment, actions, states, rewards, policy, return, discount, episodic/continuing tasks, Markov, value function, optimal value function, optimal policy, Bellman optimality equations

### 3.11 Bibliographical and Historical Remarks
Reference material is given.

# II. Elementary Solution Methods
Three methods to solve the full problem will be covered:
- Dynamic Programming: well developed but need a complete accurate model
- Monte Carlo methods: no model need and simple, but do not work for step-by-step incremental computation
- Temporal-difference learning: no model and fully incremental, but complex to analyze
Also different in efficiency and speed of convergence. In part III, they are combined to get the best of each.

## Chapter 4 Dynamic Programming
DP is a collection of algorithms used to compute optimal policies given a perfect model as an MDP. They are unfeasible because the assume a perfect model and requires too much computations. Important theoretical because the other methods try to achieve what DP does but without a perfect model and with less computation.

### 4.1 Policy Evaluation
Policy evaluation is how to compute the state-value function for a given policy.
Talks about iterative policy evaluation. Then gives examples, and uses grid world at the end. 

### 4.2 Policy Improvement
We compute the value function of a policy to find better policies. We take a different actions and then follow the found policy, and compare to see if it was better. If it is better, it is called policy improvement theorem.
Policy improvement, find a better policy from an existing policy.

### 4.3 Policy Iteration
Policy Iteration: Continue iterating with policy improvement until you find the optimal. Since it is a finite MDP, it should eventually converge. It often needs few iterations to converge.

### 4.4 Value Iteration
You do not need to reach the limit of policy evaluation, it can be stopped after the optimal policy is found.
Value iteration: policy evaluation is stopped after just one sweep (one backup of each state). It stops when it only improves by a small amount (arbitrarily set). It could be seen as an iteration of one sweep for policy evaluation and one sweep for policy improvement.

### 4.5 Asynchronous Dynamic Programming
DP needs to sweep all states. Asynchronous DP does so not in order, but with whatever state is available. Also works while working in real-time, it allows to focus on the states more relevant to the agent.

### 4.6 Generalized Policy Iteration
GPI is used to describe the interaction between policy evaluation and policy improvement. It does not need to be sequential.
Both together acheive optimality, both are opposing but collaborative.

### 4.7 Efficiency of Dynamic Programming
They are impractical for very large problems, but much better than linear programming or direct search (factor of about 100).
They typically converge way faster than their worst case scenario, in real life problem they are quite feasible.

### 4.8 Summary
Policy evaluation and policy improvement, together you get policy iteration and value iteration.
Bootstraping - update an estimate based on other estimates.
Monte Carlo - no model and no bootstrap
Temporal-difference - no model but does bootstrap

### 4.9 Bibliographical and Historical Remarks
Reference material here.

## Chapter 5 Monte Carlo Methods
First learning methods for estimating value functions and discovering optimal policies. We do not assume a complete model. Monte Carlo require experience samples, from on-line or simulated interactions. 
Value estimates and policies only change when an episode finishes.

### 5.1 Monte Carlo Policy Evaluation
Each ocurrence of a state is called a visit. First-visit MC methos averages return after the first visit. Every-visit is the other one.
The estimate of each state is independent of estimates of other states. So, no bootstrap. A third advantage is that is can learn each state independently, so it can learn only the states it is experiencing.

### 5.2 Monte Carlo Estimation of Action Values
With no model, estimate action values instead of state values, which is Q*. For this, continual exploration must be assured.

### 5.3 Monte Carlo Control
Explorinf starts is possible when you have simulated experience, you can choose a random equal probability start at all possible starts. Not necessarily possible with on-line experience.
Talks about Monte Carlo ES (Explorin Starts). No need for infinite episodes, but do policy evaluation and improvement after each episode for each action-state pait visited in each episode.

### 5.4 On-Policy Monte Carlo Control
There are on-policy methods and off-policy methods.
On-policy methods attempt to evaluate or improve the policy that is used to make decisions.

### 5.5 Evaluating One Policy While Following Another
We have experience off the policy, but we can still learn the value function for the policy from said experience. This is possible is all action in my policy is taken at least occasionally in the other policy.

### 5.6 Off-Policy Monte Carlo Control
On-policy, estimate the value of a policy while using it for control. In off-policy the two functions are separated.
- Behavior policy: to generate behavior
- Estimation policy: to evaluate and improve the value function
Advantage, estimation can be greedy, while keeping the behavior random.

### 5.7 Incremental Implementation
Gives a way to do the calculations incremental, to optimize computer resources.

### 5.8 Summary
Learn from experience, sample episodes.
Advantages over DP:
- learn from interaction with the environment
- can use simulation or sample models
- easy to focus on subset of states
- less harmed by having non Markov states, because they do not bootstrap
You need to maintain enough exploration.

### 5.9 Bibliographical and Historical Remarks
Reference material cited.

## Chapter 6 Temporal-Difference Learning
TD, central and novel idea. Combination of Monte Carlo and DP. No model needed just experience, but do bootstrap. 
In chapter 7, TD(gamma) combines TD and Monte Carlo.

### 6.1 TD Prediction
MC need to wait the end of the episode to update a value estimate, TD only needs to wait for the next time step.

### 6.2 Advantages of TD Prediction Methods
No model needed. Learn per step, not per episode. MC discards experimental actions.
In practice, but not mathematically proven, TD is faster than MC.

### 6.3 Optimality of TD(0)
Batch updating, update values after each batch of episodes or time steps (when limited you repeat them).
You have one example of state A reward 0 and move to B reward 0. Other options where B's reward is 3/4.
TD gives A's reward as 3/4, but MC gives reward 0.
TD is better because it generalizes to new data.

### 6.4 Sarsa: On-Policy TD Control
We calculate the action-value function. Move from state-action pair to state-action pair.
SARSA = State, action, reward, state, action

### 6.5 Q-Learning: Off-Policy TD Control
An important breakthrough. Q is estimated independent of which policy is followed. Only thing required is for all pairs to continue to be updated.

### 6.6 Actor-Critic Methods
Separate memory structue to represent the independence of policy and value function. 
Policy is the actor. Estimated value function is the critic. Learning is on-policy.

### 6.7 R-Learning for Undiscounted Continuing Tasks
It is off-policy, for no discounts and not episodic. Seeks the maximum reward per time step.

### 6.8 Games, Afterstates, and Other Special Cases
Afterstates, what comes after you make a move in a game. Afterstate value functions, the value functions of afterstates.
Two positions can result in the same one. Normally, they would be considered two states, but with afterstates it is considered the same, thus, faster learning.

### 6.9 Summary
Online, little computation.
Methods presented are one-step, tabular, modelfree TD methods.
Next chapter are extended to:
- Multistep forms.
- Function approximation instead of tables (neural networks)
- Also with a model.
TD methods are not reinforcement learning specific, they can also be used to learn long-term predictions about dynamical systems.

### 6.10 Bibliographical and Historical Remarks
Reference material is given.

# III. A Unified View
Three methods: DP, MC, TD. They are not exclusive.
Unify: MC and TD, function approximation, models again.

## Chapter 7 Eligibility Traces
Forward view or theoretical view: They bridge MC and TD. Understand what is computed.
Backward view or mechanistic view: An eligibility trace is a temporary record of the occurrence of an event, visit state or take action. Credit or blame are only assigned to relevant states and actions. Get intuition about the algorithm.

### 7.1 n-Step TD Prediction
TD does a one-step backup, MC steps until the end. n-step are intermmediate, more than one but not all.
Rarely used n practice, because too complicated to implement. Only theoretical purposes.

### 7.2 The Forward View of TD(gamma)
Not only n-step, also an average of n-step returns. Example, half 2 step average + half 4 step average. Called complex backup.

### 7.3 The Backward View of TD(gamma)
Useful because it is simple conceptually and computationally. Eligibility trace, additional memory variable for each state. It remembers which states and action are eligible to learn when a reinforcement event occurs.
Feasible to implement, unlike forward view.

### 7.4 Equivalence of Forward and Backward Views
It proves that off-line TG(gamma) achieves the same weight updates as the offline gamma-return algorithm.

### 7.5 Sarsa(gamma)
Use eligibility traces for control. Apply TD(gamma) to state-action pairs instead of states.

### 7.6 Q(gamma)
Combine eligibility traces and Q-learning:
- Watkin's Q(gamma)
- Peng's Q(gamma)
Watkins's Q($\lambda $) does not look ahead all the way to the end of the episode in its backup. It only looks ahead as far as the next exploratory action
Peng's Q($\lambda $) can be thought of as a hybrid of Sarsa($\lambda $) and Watkins's Q($\lambda $).

### 7.7 Eligibility Traces for Actor-Critic Methods
Shows how to add eligibility traces to actor-critic methods

### 7.8 Replacing Traces
A replacing trace, when a visit is repeated, the max is 1, does not go beyond.

### 7.9 Implementation Issues
Only recent visits need to be updated, as they are the only ones greater than 0.

### 7.10 Variable Lambda
The idea of changing lambda from step to step.

### 7.11 Conclusions
Eligibility traces with TD errors are an incremental way of shifting between MC and TD.
Eligibility traces are used for delayed rewards and non-Markov states.
They require more computation than one-step methods, but faster learning, especially when rewards are delayed by many steps. 

### 7.12 Bibliographical and Historical Remarks
Reference material presented here.

## Chapter 8 Generalization and Function Approximation
Generalization problem, learn from previous similar (not exact) states.
Function approximation, it is supervised learning.

### 8.1 Value Prediction with Function Approximation
Values function is not a table, but a function, like a neural network weight matrix.
Any supervised learning method that can learn with online data works.

### 8.2 Gradient-Descent Methods
Particularly well suited for reinforcement learning.
2 methods have been used:
- Multilayer ANNs with backprop
- Linear form

### 8.3 Linear Methods
V is a linear function of theta.

#### 8.3.1 Coarse Coding
Inside a circle or not, example of binary feature. Circles overlap, so by knowing on which circles a state is present, you can coarsely know its location. Like an estimate, called coarse coding.

#### 8.3.2 Tile Coding
Form of coarse coding, for on-line learning.

#### 8.3.3 Radial Basis Functions
Natural generalization of coarse coding. Not only values 0 and 1, but anything between them, real numbers. 
RBF network - linear function approximator.
More complex and more manual tuning is needed.

#### 8.3.4 Kanerva Coding
With hundreds of dimensions, RBF and tile coding are impractical.
Separate complexity of states from the complexity of approximations. One method is Kanerva coding.

### 8.4 Control with Function Approximation
Now action-state prediction, combine policy improvement and action selection.
It shows a mountain-car task problem which is somewhat interesting in that it has to purposely get worse before it can get better to reach the final goal.

### 8.5 Off-Policy Bootstrapping
Interaction between bootstrapping, function approximation and on-policy distribution.
Bootstrapping methods are harder to combine with function approximation than non-bootstrapping. The combination is unstable.

### 8.6 Should We Bootstrap?
Non-bootstraping seem to be more convenient theoretically, but bootstraping methods are the methods of choice in practice.
Empirically, bootstraping methods perform better, not yet known why (when book was written).

### 8.7 Summary
Reinforcement learning methods need generalization, supervised-learning methods help in this.

### 8.8 Bibliographical and Historical Remarks
Reference material.

## Chapter 9 Planning and Learning
Unify methods that require a model and those that don't.
- Planning methods: need a model
- Learning methods: do not need a model

### 9.1 Models and Planning
With a model you can simulate the environment. Planning, input is a model, output is a policy (improved preferrably).
state-space planning, search the possible states
plan-space planning, search the possible plans

### 9.2 Integrating Planning, Acting, and Learning
Model-learning: improve the model, make it match closer to reality
Direct reinforcement learning: improve value function and policy
value/policy -acting-> experience -model learning-> model -planning-> value/policy
value/policy -acting-> experience -direct RL-> value/policy
The complete loop is also called indirect RL.
Indirect need less experience. Direct are simpler.
Planning seems to make learning faster (in number of episodes required). They use Dyna-Q agents.
In a maze example, no planning 25 episodes, 5 plan steps 5 episodes, 50 plan steps 3 episodes.
Real experience for learning, simulated experience for planning. Both actions (learning and planning) happen simultaneously.

### 9.3 When the Model is Wrong
When wrong, planning will compute a sub-optimal policy.
When the optimal path at the start is long, and after learning it a new short path opens, planning makes it so that it never finds the new path. Only Dyna-Q+ found it.
Dyna-Q+ keeps track of how long ago did it try a state, and explores the oldest.
reward+k(square_root(n)), where n is time steps since last try, for a small k. 
It permits to try the mentioned steps, at the same time as planning to try them, if they have a long sequence. Normal epsilon-greedy will most likely never reach a state randomly if a very long sequence is required.

### 9.4 Prioritized Sweeping
Planning has been done with random searches. On larger problems, it is inefficient.
Move backwards from goal states(any state whose value has changed), prioritized by size of change.

### 9.5 Full vs. Sample Backups
Full is better but much more expensive computationally, which is the limiting resource.
Sample are better where large stochastic problems with too many states, because they compound their learning little by little.

### 9.6 Trajectory Sampling
Two ways of distributing backups, classical from DP (sweeps of entire state space), backup up once per sweep, not good for large tasks, one sweep may be too much. 
Sample from state space with some distribution. Trajectory samplig, generate experience and backups to create said distribution. For small problems, better at the start, worse later on. This is because it keeps exploring already optimal states. Results not conclusive.

### 9.7 Heuristic Search
They are the predominant planning methods. Planning in the policy part. Many possible actions are evaluated, backup up, and choses the best one, discarding the others. Deeper the search, better actions but more computation and time. 

### 9.8 Summary
Learning and planning, let them update the same estimated value function.
Any learning methods becomes a planning methods by applying it to a simulated experience.
Sometimes deep backups can be implemented as sequences of shallow backups.

### 9.9 Bibliographical and Historical Remarks
Reference material.

## Chapter 10 Dimensions of Reinforcement Learning
Not a collection of individual methods, a cohorent set of ideas (dimensions).

### 10.1 The Unified View
All methods have 3 key ideas;
- estimate value function is their goal
- backup values
- GPI, keep an approximate value function and policy, and one improves the other

2 dimensions:
- sample or full backups
- shallow or deep backups

- DP: full and shallow
- TD: sample and shallow
- MC: sample and deep
- Exhaustive search: full and deep

another dimension is function approximation. another is needing or not a model. another is on or off policy.

More dimensions:
- definition of return: episodic or continuing, discounted or undiscounted
- action values vs state values vs afterstate values
- action selection / exploration
- synchronous vs asynchronous: backups of states simultaneous or not
- replacing vs accumulating traces
- real vs simulated
- location of backups
- timing of backups: as part of selections actions or later
- memory of backups: how long to keep them

### 10.2 Other Frontier Dimensions
Mentioned current work to be done for the advancement of reinforcement learning.

## Chapter 11 Case Studies
Show some case studies where reinforcement learning has been applied.

### 11.1 TD-Gammon
Made to play backgammon. Used a multilayer neural network in 1992. Learned to play at the same level of the best players in the world. 

### 11.2 Samuel's Checkers Player
1959. Heuristic search methods and TD.
It sometimes got worse with experience, it was missing evaluation functions. It acheived better than average play. 

### 11.3 The Acrobot
A bot swinging on a high bar, like a gymnast.
Sarsa(lambda) was the algorithm used.

### 11.4 Elevator Dispatching
Press the elevator button and wait for it. Wait time depends on dispatching strategy.
4 elevators, 10 floors, was analyzed. 10^22 states. If 1 per microsecond, we need 1000 years per sweep.
Waiting time: time to get on
System time: time to get off
objective is focused on squared waiting time.
one-step q-learning

### 11.5 Dynamic Channel Allocation
How to use bandwidth in cellular phone system, to provide good service to as many customer as possible. The RL performed a little better, goal was to reduces amount of blocked calls.

### 11.6 Job-Shop Scheduling
Tasks and resources. A resources can only do one task at once. Tasks have dependencies within each other. Goal is to minimize time to execute all tasks. 
Experience replay technique developed by Lin(1992). At any point in learning it always remembered the best episode up to that point, after every four episodes it replayed this remembered episode, learning again from it.
First application in plan-space. To search for the best plan.

## Bibliography
References used in the book.