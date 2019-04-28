# Reinforcement-Learning-Sequence-Alignment

Attempt at aligning two sequences using reinforcement learning. Please feel free to try out or tweak
the sequence alignment program. You can even develop your own novel solution to the enviroment.

## What is alignment?
![alt text](https://www.mesquiteproject.org/files/multiSplitBefore.gif "Sequence Alignment")


To align two sequences means to insert dashes at specific indices,
shifting the overall sequence, to ensure that the two sequences match the most 
as a whole. Another way to think about it is as a score-maximizing problem,
where different rewards or penalties are imposed for matching, mismatching,
or dashes.

Learn more about alignment here: https://en.wikipedia.org/wiki/Sequence_alignment.

Alignment is applicable particularly in the field of bioinformatics, where 
researchers may want to determine the evolutionary relatedness of two organisms.
Another application is in next generation sequencing, where researchers 
want to determine overlapping sequences.

## Overview

This project takes a reinforcement learning approach to aligning two sequences.
An environment is created, to provide new sequences to the user.
The style of the environment mimics (and was inspired by) 
that of OpenAI gyms (http://gym.openai.com/envs/).

## Environment

### Representation
Following is the representation of elements in the sequence

*   -1 - Dash (-)
*   1 - Adenine (A)
*   2 - Thymine (T)
*   3 - Cytosine (C)
*   4 - Guanine (G)


### Actions

*   0 - no dash
*   1 - dash on seq1
*   2 - dash on seq2

### Rules

The objective of the sequence alignment game is to get the highest score through making decisions. In this game, there are two sequences of nucleotides (like a four letter alphabet).The game ends when the index reaches the end of the last sequence, and at that point the final scores are tallied.

### Score

Score is computed as follows: for each index in the sequence, 1 is added to the sequence if the two are matching, -1 is added if there is a mismatch, and -2 is added if there is a gap in one of the sequences

### Variables

*   index - represents the current index the game is currently at
*   done - represents whether or not the game is over
*   seq1 - one nucleotide sequence to be aligned
*   seq2 - another nucleotide sequence to be aligned
*   observation - 1x3 array (index, seq1, seq2)

### Methods

*   reset - returns a fresh new game, resetting the index and making new sequences
*   step - given an action to take, returns observation, reward, and whether game is done
*   display - displays the two sequences and color-coded current index
*   randomAction - returns a random integer between 1 and TOTALACTIONS (inclusive), representing a random action
*   replay - replay

## Solution Approach

### Overview of Q Learning

Q Learning is a type of reinforcement learning that seeks to optimize the action taken in a particular state.
This value it attempts to learn is called Q, hence the name "Q Learning"
Deep Q learning is a type of Q-learning that applies deep neural networks in the prediction of Q.
The solution taken uses Q Learning to predict the value of taking an action given a particular state.
To clarify, state refers to information gleaned from an observation, which influences the action taken.

Here is how the computer learns:
1) Taking Random actions - First, the program takes a series of random actions in any given state, and simply 
observes the reward. The rationale behind doing this is that information of the reward in each particular 
state will be used to train the computer to accurately predict the value of Q in each given state.

2) Experience Replay - Here is the actual "learning" part of reinforcement learning. The computer then looks back at
all the random actions it took. The particular state at which the computer took the action becomes the input, 
and the reward of taking that action (provided by the environment) becomes the output. The computer then goes back 
to this data and trains itself on it, until it has a good sense of the value of taking a particular action in a particular state.
It is for this reason that Q is sometimes viewed as a function of both state and action -- Q(S, A)

3) Testing - By this point, the computer has learned the value of taking an action in each particular state. The hope is 
that it can then use this knowledge to identify the optimal action to take in any given state. At this point, the user
can plug and play, identifying how the program performs when presented with different sequences, and determining the 
optimal action to take in each.

### Model Architecture

The architecture of this model consists of a series of dense layers, with the output layer consisting of the Q values
for each of the three possible actions, outlined above.

model = Sequential()
model.add(Dense(20, input_shape=(201,), init='uniform', activation='relu'))
model.add(Dense(18, init='uniform', activation='relu'))
model.add(Dense(10, init='uniform', activation='relu'))
model.add(Dense(env.TOTALACTIONS, init='uniform', activation='linear'))

These simple lines outline the architecture of the model, which is discussed further below.

* Layer 1 (Dense Layer) - This layer is provided with the two sequences that are to be aligned, as well as the current index.
The output of this layer is an array of the shape (*, 20), with relu activation and uniform initialization of weights.
* Layer 2 (Dense Layer) - Another dense layer, with an output array shape (*, 18) - relu activation, uniform initialization of weights
* Layer 3 (Dense Layer) - Another dense layer, with an output array shape (*, 10) - relu activation, uniform initialization of weights
* Layer 4 (Output Layer) - Another dense layer, with an output array shape (*, env.TOTALACTIONS) - relu activation, uniform initialization of weights
In our case, there are three possible actions, so the output will be of shape (*, 3)

### Performance

Unfortunately, the program did not display optimal performance, even after adequate training. Here are some potential reasons.

* Creation of Data - The creation of the data is random, and starts from scratch; this does not accurately mimic actual alignment problems,
nor does it allow for effective training
* Reward system - The reward returned by the environment for each particular action is calculated as the difference of score
before and after taking the action. The calculation of the score is commonly used even by other algorithms, but there may be 
better alternatives to calculating reward, which is currently very short-term.
* Model architecture - Perhaps a different model architecture, such as one involving Recurrent Neural Networks (RNN) would
better serve the purposes of this solution.
