# Mathematical Formulation

## 1. Problem setup

The environment is a **partially observable Markov decision process (POMDP)**. The simulator has a complete state \(s_t\), but the agent only receives a limited egocentric observation \(o_t\).

\[
\mathcal{M}=(\mathcal{S},\mathcal{A},\mathcal{O},T,\Omega,r,\gamma,b_0)
\]

where:

- \(s_t \in \mathcal{S}\): true environment state, including agent position and orientation, goal location, and distractor locations
- \(a_t \in \mathcal{A}\): action taken by the agent
- \(o_t \in \mathcal{O}\): current 7×7 egocentric observation
- \(T(s_{t+1}\mid s_t,a_t)\): environment transition model
- \(\Omega(o_t\mid s_t)\): observation model determining what part of the state is visible
- \(r(s_t,a_t)\): task reward, hidden from the IRL learner
- \(\gamma\): discount factor

The important distinction is:

\[
\boxed{s_t \neq o_t}
\]

The simulator knows \(s_t\), but the partial-observation agent does not. Multiple underlying world states can produce the same current observation.

---

## 2. Ideal POMDP solution: maintain a belief

Because \(o_t\) does not uniquely determine \(s_t\), an optimal POMDP agent reasons using a **belief state**:

\[
\boxed{
b_t(s)=P(s_t=s\mid o_{1:t},a_{1:t-1})
}
\]

The belief is a probability distribution over the possible states of the world given everything observed so far.

Conceptually:

\[
b_{t-1}
\xrightarrow[\text{action }a_{t-1}]{\text{predict}}
P(s_t\mid o_{1:t-1},a_{1:t-1})
\xrightarrow[\text{observation }o_t]{\text{Bayes update}}
b_t
\]

The complete Bayesian update can be written as

\[
b_t(s')
\propto
\Omega(o_t\mid s')
\sum_s
T(s'\mid s,a_{t-1})b_{t-1}(s)
\]

or more compactly,

\[
\boxed{
b_t=\tau(b_{t-1},a_{t-1},o_t)
}
\]

where \(\tau\) is the Bayesian belief-update operator.

The purpose of the belief is to compress the entire observation/action history into the information about the world that matters for future decisions.

---

## 3. Current implementation: learned recurrent state instead of explicit belief

The current model does **not** explicitly calculate \(b_t\).

Instead, the observation is first encoded:

\[
x_t=f_\psi(o_t)
\]

where \(f_\psi\) is either a CNN or MLP featurizer.

A GRU then integrates the current observation with previous history:

\[
\boxed{
h_t=
\operatorname{GRU}_\phi(x_t,h_{t-1})
}
\]

with

\[
h_0=0.
\]

Therefore the implemented perception pipeline is:

\[
\boxed{
o_t
\rightarrow
f_\psi(o_t)
\rightarrow
h_t
\rightarrow
Q_\theta(h_t,a)
}
\]

More explicitly:

\[
\text{egocentric observation}
\rightarrow
\text{CNN/MLP features}
\rightarrow
\text{GRU history representation}
\rightarrow
\text{Q-values}
\rightarrow
\text{action}
\]

The important distinction is that

\[
\boxed{
h_t \neq b_t
}
\]

in general.

A true belief \(b_t\) is an explicit probability distribution over possible states. The GRU hidden state \(h_t\in\mathbb{R}^d\) is instead a learned deterministic representation of observation history.

The working assumption is therefore not that the GRU exactly performs Bayesian filtering, but that it can learn a representation containing enough belief-relevant information for good decisions:

\[
\boxed{
(o_{1:t},a_{1:t-1})
\longrightarrow
h_t
\approx
\text{decision-relevant representation of }b_t
}
\]

This distinction is important because an explicit belief has persistent probabilistic semantics, while the GRU has no built-in guarantee that information such as a previously observed goal location will remain encoded over long sequences.

---

# 4. IQ-Learn operates on the recurrent representation

The recurrent state \(h_t\) is passed into a soft Q-function:

\[
Q_\theta(h_t,a)
\]

which estimates the long-term value associated with each action given the information currently represented by \(h_t\).

IQ-Learn does not separately train a policy network. Instead, the learned Q-function implies a stochastic policy:

\[
\boxed{
\pi_\theta(a\mid h)
=
\frac{\exp(Q_\theta(h,a)/\alpha)}
{\sum_{a'}\exp(Q_\theta(h,a')/\alpha)}
}
\]

where \(\alpha\) controls the softness of the policy.

The corresponding soft value function is

\[
\boxed{
V_\theta(h)
=
\alpha
\log
\sum_a
\exp\left(\frac{Q_\theta(h,a)}{\alpha}\right)
}
\]

which is the entropy-regularized analogue of

\[
V(h)=\max_a Q(h,a).
\]

As \(\alpha\rightarrow0\),

\[
V_\theta(h)\rightarrow\max_a Q_\theta(h,a).
\]

---

# 5. Recovering reward from the Q-function

The ordinary Bellman relationship is

\[
Q(s,a)
=
r(s,a)
+
\gamma
\mathbb{E}[V(s')]
\]

which can be rearranged as

\[
r(s,a)
=
Q(s,a)
-
\gamma
\mathbb{E}[V(s')].
\]

IQ-Learn exploits this relationship in the inverse direction. Rather than learning a separate reward network and then training a policy against it, it learns a Q-function from expert demonstrations.

In the recurrent implementation, the corresponding recovered reward signal is

\[
\boxed{
\hat r(h_t,a_t)
=
Q_\theta(h_t,a_t)
-
\gamma V_\theta(h_{t+1})
}
\]

so the learned Q-function simultaneously determines:

\[
Q
\rightarrow
\begin{cases}
\pi(a\mid h) & \text{behavior}\\
\hat r(h,a) & \text{implied reward}
\end{cases}
\]

Importantly, the learner receives expert demonstrations but **does not receive the simulator's true task reward during IQ-Learn training**.

---

# 6. Complete implemented pipeline

Putting the pieces together:

\[
\boxed{
s_t
\xrightarrow{\Omega}
o_t
\xrightarrow{f_\psi}
x_t
\xrightarrow{\operatorname{GRU}_\phi(\cdot,h_{t-1})}
h_t
\xrightarrow{Q_\theta}
Q_\theta(h_t,\cdot)
\xrightarrow{\operatorname{softmax}}
\pi(a_t\mid h_t)
}
\]

The roles of each component are:

| Component | Meaning |
|---|---|
| \(s_t\) | Actual state of the MiniGrid world |
| \(\Omega\) | Restricts the state to the agent's egocentric view |
| \(o_t\) | What the agent currently sees |
| \(f_\psi\) | Extracts features from the current observation |
| \(h_t\) | Recurrent representation of observation history |
| \(Q_\theta(h_t,a)\) | Learned long-term value of each action |
| \(\pi(a_t\mid h_t)\) | Policy implied by the learned Q-values |

The theoretical POMDP pipeline would instead be:

\[
\boxed{
s_t
\rightarrow
o_t
\rightarrow
b_t
\rightarrow
Q(b_t,a)
\rightarrow
a_t
}
\]

while the current approximation is:

\[
\boxed{
s_t
\rightarrow
o_t
\rightarrow
h_t
\rightarrow
Q(h_t,a)
\rightarrow
a_t.
}
\]

The central representational question is therefore:

\[
\boxed{
\text{Does }h_t\text{ retain the information about }b_t
\text{ needed for reward-directed behavior?}
}
\]

---

# 7. Why reward maximization can produce active sensing

For an explicit belief state, define the expected immediate reward as

\[
\bar r(b,a)
=
\mathbb{E}_{s\sim b}[r(s,a)].
\]

The optimal belief-state value satisfies

\[
\boxed{
V^*(b)
=
\max_a
\left[
\bar r(b,a)
+
\gamma
\mathbb{E}_{o'\sim p(o'|b,a)}
V^*(\tau(b,a,o'))
\right].
}
\]

This equation is central to the hypothesis.

There is **no explicit information-gain reward**.

However, an action affects which observation the agent receives:

\[
a_t
\rightarrow
o_{t+1}
\]

which affects the updated belief:

\[
o_{t+1}
\rightarrow
b_{t+1}
\]

which affects future decisions:

\[
b_{t+1}
\rightarrow
a_{t+1}
\]

and therefore future task reward.

The causal chain is:

\[
\boxed{
\text{sensing action}
\rightarrow
\text{informative observation}
\rightarrow
\text{better belief}
\rightarrow
\text{better future action}
\rightarrow
\text{greater future reward}
}
\]

Therefore, an information-seeking action can have positive long-term value even when

\[
r_{\text{information}}=0.
\]

This is the sense in which **active sensing could emerge instrumentally from ordinary reward maximization**.

---

# 8. Independent information-seeking baseline

To test whether reward maximization produces behavior resembling explicit information seeking, define uncertainty over the belief using Shannon entropy:

\[
\boxed{
H(b)
=
-\sum_s b(s)\log b(s)
}
\]

An action's expected information gain is

\[
\boxed{
IG(b,a)
=
H(b)
-
\mathbb{E}_{o'\sim p(o'|b,a)}
\left[
H(\tau(b,a,o'))
\right].
}
\]

An information-only policy would choose

\[
\boxed{
a_I^*
=
\arg\max_a IG(b,a).
}
\]

The reward-only policy instead chooses according to its learned value:

\[
\boxed{
a_R^*
=
\arg\max_a Q(h,a).
}
\]

Critically, information gain is **never included in the reward-only policy's training objective**.

The experimental question is therefore whether

\[
a_R^*
\]

systematically aligns with

\[
a_I^*
\]

in states where gathering information is useful.

If so, information-seeking behavior has emerged from reward-directed learning rather than being explicitly rewarded.

---

# 9. Why distractors matter

The environment contains both a task-relevant goal \(G\) and reward-irrelevant distractors \(D\).

This allows a stronger distinction than simply asking whether the agent explores.

A generic uncertainty-reduction agent may value information about either:

\[
IG(G;o'\mid b,a)
\]

or

\[
IG(D;o'\mid b,a).
\]

A reward-directed agent should preferentially value information when it improves decisions relevant to the goal.

The stronger hypothesis is therefore:

\[
\boxed{
\text{reward maximization produces selective, task-relevant information seeking}
}
\]

rather than simply

\[
\text{reward maximization produces exploration}.
\]

---

# 10. What the current results tell us about the representation

Performance improved strongly with demonstration count:

\[
200\rightarrow400\rightarrow800\rightarrow1600
\]

demonstrations produced approximately

\[
60\%\rightarrow70\%\rightarrow79\%\rightarrow82\%
\]

success.

However, the 1,600-demonstration model remained highly path-inefficient:

\[
\text{mean path length}=73.5
\]

versus

\[
\text{BFS optimal}=6.5,
\]

giving

\[
\boxed{
\text{path ratio}\approx11.9\times.
}
\]

Only approximately

\[
7\%
\]

of trajectory steps occurred before the goal was first observed.

Therefore, the main inefficiency does not appear to come from simply failing to find the goal initially. Most behavior occurs **after the agent has already observed it**.

A current hypothesis is therefore:

\[
\boxed{
\text{goal observed}
\rightarrow
h_t\text{ initially encodes goal information}
\rightarrow
\text{goal leaves view}
\rightarrow
\text{goal information may degrade in }h_t.
}
\]

This has not yet been directly demonstrated. A natural test is to decode goal position from \(h_t\) as a function of time since the goal was last visible.

---

# 11. Current interpretation and next step

The current architecture tests whether a generic recurrent representation can learn enough implicit state information to support reward-directed behavior under partial observation.

The next stage should separate two questions:

### Representation

Can the agent maintain a useful estimate of the hidden world state?

\[
H_t\rightarrow b_t
\]

### Decision-making

Given that belief, does reward maximization itself produce information-seeking actions?

\[
b_t
\rightarrow
Q(b_t,a)
\rightarrow
a_t
\]

An explicit belief representation would make uncertainty directly measurable and allow the information-seeking comparison to be defined cleanly:

\[
H(b_t),\qquad IG(b_t,a).
\]

This gives the overall research pipeline:

\[
\boxed{
\text{partial observation}
\rightarrow
\text{belief / history representation}
\rightarrow
\text{reward-directed value}
\rightarrow
\text{behavior}
}
\]

with the central experimental question:

\[
\boxed{
\textbf{Can reward-directed learning alone produce selective actions that gather task-relevant information?}
}
\]