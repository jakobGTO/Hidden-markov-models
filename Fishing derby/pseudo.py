'''
def init_parameters(self):

    Initialize Matrices
    Create List with N_Species Sub-Lists to store HMM models -> List_HMM
    Create List with N_Fish sub-lists to store observation sequence for that specific fish -> List_obs
    Create List of already observed fishes -> List_fish

def guess(self, step, observations):
    # Split observation data into List_obs ( function already created in player.py, it's called split_observation)

    if (step == 1 ):
        # Make random guess
    else:
        probabilities = []       # Store sum of probabilities for each fish species
        current_fish = first available fish that is not on the observed fish list (List_fish)
        for i in range(N_Species):
            suma = 0            # sum of probabilities
            for x in range(len(List_HMM[i])):       # Go through all HMMs for fish_type i
                suma += forward( HMM = x, seq = self.obs[self.fish_number] )       # Run forward algorithm with sequence observed for the current fish
            scaled_sum = suma / len(List_HMM[i]) <- Not sure if this is the correct way to do it, but it needs to be scaled
            probabilities.append(suma)
        
        Guess = ( current_fish , argmax(probabilities))       # First guesses will be rubish but as we get increasingly more HMMs it will get better


def reveal(self, correct, fish_id, true_type):
    HMM = Baum-Welch()
    List_HMM[true_type].append(HMM) #Store HMM to respective position (fish index)
    
    
---------------- Approach 2 ------------------------

def init_parameters(self):

    Initialize Matrices
    Create List with N_Emissions Sub-Lists to store HMM models -> List_HMM
    Create List with N_Fish sub-lists to store observation sequence for that specific fish -> List_obs
    Create List of already observed fishes -> List_fish

def guess(self, step, observations):
    # Split observation data into List_obs ( function already created in player.py, it's called split_observation)

    if (step == 1 ):
        # Make random guess
    else:
        probabilities = []       # Store sum of probabilities for each fish species
        current_fish = first available fish that is not on the observed fish list (List_fish)
        for i in range(N_Species):
            suma = 0            # sum of probabilities
            for x in range(len(List_HMM[i])):       # Go through all HMMs for fish_type i
                suma += forward( HMM = x, seq = self.obs[self.fish_number] )       # Run forward algorithm with sequence observed for the current fish
            scaled_sum = suma / len(List_HMM[i]) <- Not sure if this is the correct way to do it, but it needs to be scaled
            probabilities.append(suma)
        
        Guess = ( current_fish , argmax(probabilities))       # First guesses will be rubish but as we get increasingly more HMMs it will get better


def reveal(self, correct, fish_id, true_type):
    HMM = Baum-Welch()
    List_HMM[true_type].append(HMM) #Store HMM to respective position (fish index)
    
'''
