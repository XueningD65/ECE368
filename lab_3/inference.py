import numpy as np
import graphics
import rover

def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """

    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    forward_messages[0] = prior_distribution
    backward_messages = [None] * num_time_steps
    marginals = [None] * num_time_steps

    # Set up the distribution

    # Forward messages
    #print("Forward")
    for idx, msg in enumerate(forward_messages):
        forward_messages[idx] = rover.Distribution({})

        for state in all_possible_hidden_states:
            # observation missing
            if observations[idx] is None:
                cond_prob = 1
            else:
                cond_prob = observation_model(state)[observations[idx]]
            prior = prior_distribution[state]

            if idx == 0: # initialization
                if cond_prob != 0 and prior != 0: # only include non-zero states
                    forward_messages[0][state] = cond_prob * prior
            else:
                sum = 0
                #print(forward_messages[idx - 1])
                for prev_state in forward_messages[idx - 1].keys():
                    prob = forward_messages[idx - 1].get(prev_state)
                    sum += prob * transition_model(prev_state)[state]
                #print(sum)
                if sum * cond_prob != 0: # only include non-zero states
                    forward_messages[idx][state] = sum * cond_prob

        forward_messages[idx].renormalize()
        #print("idx:", idx, "info:", forward_messages[idx])

    #Backward message
    #print("Backward")
    for offset in range(num_time_steps):
        idx = num_time_steps - offset - 1
        backward_messages[idx] = rover.Distribution({})
        for state in all_possible_hidden_states:
            # initialization
            if offset == 0:
                backward_messages[idx][state] = 1
            else:
                sum = 0
                for next_state in backward_messages[idx + 1].keys():

                    if observations[idx + 1] is None: # missing observation
                        cond_prob = 1
                    else:
                        cond_prob = observation_model(next_state)[observations[idx + 1]]

                    sum += cond_prob * backward_messages[idx + 1].get(next_state) * transition_model(state)[next_state]

                if sum != 0: # only non-zero states
                    backward_messages[idx][state] = sum

        backward_messages[idx].renormalize()
        #print("idx:", idx, "info:", backward_messages[idx])

    # Marginal distribution
    for idx in range(num_time_steps):
        marginals[idx] = rover.Distribution({})

        sum = 0
        for state in all_possible_hidden_states:
            if forward_messages[idx][state] * backward_messages[idx][state] != 0:
                marginals[idx][state] = forward_messages[idx][state] * backward_messages[idx][state]
                sum += forward_messages[idx][state] * backward_messages[idx][state]

        for key in marginals[idx]:
            marginals[idx][key] /= sum

    return marginals

def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """

    # initialization
    num_time_steps = len(observations)
    w = [None] * num_time_steps
    local_winner = [None] * num_time_steps
    estimated_hidden_states = [None] * num_time_steps

    # calculate W
    for idx in range(num_time_steps):
        w[idx] = rover.Distribution({})
        local_winner[idx] = {}

        for state in all_possible_hidden_states:
            if observations[idx] is None: # missing observation
                cond_prob = 1 # log(1) = 0
            else:
                cond_prob = observation_model(state)[observations[idx]]

            prior = prior_distribution[state]

            if idx == 0:  # initialization
                if cond_prob != 0 and prior != 0:  # only include non-zero states
                    w[idx][state] = np.log(cond_prob) + np.log(prior)

            else:
                max_state = float("-inf")

                for prev_state in w[idx - 1].keys():
                    if transition_model(prev_state)[state] != 0 and cond_prob != 0:
                        # if a step with higher probability
                        if np.log(transition_model(prev_state)[state]) \
                                        + w[idx - 1].get(prev_state) > max_state:
                            max_state = max(max_state, np.log(transition_model(prev_state)[state]) \
                                            + w[idx - 1].get(prev_state))
                            # record the path
                            local_winner[idx][state] = prev_state

                if cond_prob != 0:
                    w[idx][state] = np.log(cond_prob) + max_state

        #print("idx:", idx, "info:", w[idx])

    # backtracking
    # find the max key and element in the last step
    max_key = max(w[num_time_steps - 1], key = w[num_time_steps - 1].get)
    max_state = w[num_time_steps - 1].get(max_key)
    estimated_hidden_states[num_time_steps - 1] = max_key

    print(max_key, max_state)

    curr_step = num_time_steps - 2
    prev_key = max_key
    while curr_step >= 0:
        estimated_hidden_states[curr_step] = local_winner[curr_step + 1][prev_key]
        prev_key = estimated_hidden_states[curr_step]
        curr_step -= 1

    #print(estimated_hidden_states)

    return estimated_hidden_states

if __name__ == '__main__':
   
    enable_graphics = False
    
    missing_observations = True
    if missing_observations:
        filename = 'test_missing.txt'
    else:
        filename = 'test.txt'
            
    # load data    
    hidden_states, observations = rover.load_data(filename)
    num_time_steps = len(hidden_states)

    all_possible_hidden_states   = rover.get_all_hidden_states()
    all_possible_observed_states = rover.get_all_observed_states()
    prior_distribution           = rover.initial_distribution()

    print("all possible hidden states")
    print(all_possible_hidden_states)

    print("all possible observed states")
    print(all_possible_observed_states)

    print("Prior distribution")
    print(prior_distribution)

    print("transition model")
    print(rover.transition_model)

    print("observations")
    print(observations)

    
    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 rover.transition_model,
                                 rover.observation_model,
                                 observations)
    print('\n')

    # sanity check
    print(marginals[1])

    timestep = num_time_steps - 1
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    print("Most likely parts of marginal at time %d:" % (30))
    print(sorted(marginals[30].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    print('\n')

    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])

    # Forward-backward and viterbi error
    forward_backward_err, viterbi_err = 0, 0
    forward_backward_cnt, viterbi_cnt = 0, 0

    for i in range(num_time_steps):
        # correct counter of viterbi method
        if estimated_states[i] == hidden_states[i]:
            viterbi_cnt += 1
        else:
            print("viterbi index", i, "move", estimated_states[i], "should be", hidden_states[i])


        # correct counter of forward-backward
        max_key = max(marginals[i], key = marginals[i].get)
        if max_key == hidden_states[i]:
            forward_backward_cnt += 1
        else:
            print("forward-backward index", i, "move", max_key, "should be", hidden_states[i])

        if i == 64:
            print("forward-backward index", 64, "move", max_key, "should be", hidden_states[64])

    forward_backward_err = (num_time_steps - forward_backward_cnt)
    viterbi_err = (num_time_steps - viterbi_cnt)
    print("forward-backward error:", forward_backward_err)
    print("viterbi error:", viterbi_err)

    # if you haven't complete the algorithms, to use the visualization tool
    # let estimated_states = [None]*num_time_steps, marginals = [None]*num_time_steps
    # estimated_states = [None]*num_time_steps
    # marginals = [None]*num_time_steps
    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()

