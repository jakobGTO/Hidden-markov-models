#!/usr/bin/env python3

'''
    ------------------------------------  Solution for Fishing Derby : HMM -------------------------------------------------------------

                                        KTH Royal Institute of Technology
                                            M.Sc Machine Learning 20/21

                                        DD280 - Artificial Intelligence

                                        Diogo Pinheiro & Jakob Lindén
                                        
                                        
                                        
        Overview:
            1) Wait 20 sec to get more data
            2) On the 20th sec, make a random guess
            3) With that random guess we can obtain the true type. Then we train the HMM (Baum-Welch Algorithm) with the 
               observation sequence of the species
            4) Assign HMM to list in which the index is the species
            5) After the 20th sec, we make prediction using the Forward Algorithm, which will go through the HMMs in all species

    -------------------------------------------------------------------------------------------------------------------------
'''

from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import random
from Baum_Welch import run_baum_welch, forward_descaled, forward_guess
import copy


def split_observation(res, obs):
    for i, x in enumerate(obs):
        res[i].append(x)
    return res


def getFish(observedFish):
    for i in range(70):
        if i not in observedFish:
            return i


class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """

        # Initialize matrices
        self.q = [0.14, 0.15, 0.135, 0.1315, 0.16, 0.13, 0.1535]
        self.A = [[0.15, 0.1535, 0.16, 0.14, 0.135, 0.13, 0.1315], [0.14, 0.16, 0.1315, 0.15, 0.13, 0.135, 0.1535], [0.14, 0.1315, 0.135, 0.15, 0.1535, 0.16, 0.13], [0.15, 0.13, 0.16, 0.1315, 0.1535, 0.14, 0.135],
                  [0.15, 0.1315, 0.14, 0.135, 0.16, 0.1535, 0.13], [0.13, 0.1315, 0.15, 0.1535, 0.14, 0.16, 0.135], [0.14, 0.1315, 0.135, 0.15, 0.13, 0.16, 0.1535]]
        self.B = [[0.125, 0.120, 0.115, 0.130, 0.135, 0.1225, 0.1225, 0.130], [0.115, 0.125, 0.1225, 0.12, 0.135, 0.13, 0.13, 0.1225], [0.1225, 0.1225, 0.13, 0.115, 0.13, 0.12, 0.135, 0.125], [0.115, 0.13, 0.125, 0.1225, 0.135, 0.1225, 0.13, 0.12],
                  [0.13, 0.125, 0.115, 0.12, 0.1225, 0.13, 0.135, 0.1225], [0.125, 0.13, 0.1225, 0.135, 0.13, 0.12, 0.1225, 0.115], [0.115, 0.13, 0.12, 0.1225, 0.1225, 0.13, 0.125, 0.135]]

        self.q_init = [0.14, 0.15, 0.135, 0.1315, 0.16, 0.13, 0.1535]
        self.A_init = [[0.15, 0.1535, 0.16, 0.14, 0.135, 0.13, 0.1315], [0.14, 0.16, 0.1315, 0.15, 0.13, 0.135, 0.1535], [0.14, 0.1315, 0.135, 0.15, 0.1535, 0.16, 0.13], [0.15, 0.13, 0.16, 0.1315, 0.1535, 0.14, 0.135],
                       [0.15, 0.1315, 0.14, 0.135, 0.16, 0.1535, 0.13], [0.13, 0.1315, 0.15, 0.1535, 0.14, 0.16, 0.135], [0.14, 0.1315, 0.135, 0.15, 0.13, 0.16, 0.1535]]
        self.B_init = [[0.125, 0.120, 0.115, 0.130, 0.135, 0.1225, 0.1225, 0.130], [0.115, 0.125, 0.1225, 0.12, 0.135, 0.13, 0.13, 0.1225], [0.1225, 0.1225, 0.13, 0.115, 0.13, 0.12, 0.135, 0.125], [0.115, 0.13, 0.125, 0.1225, 0.135, 0.1225, 0.13, 0.12],
                       [0.13, 0.125, 0.115, 0.12, 0.1225, 0.13, 0.135, 0.1225], [0.125, 0.13, 0.1225, 0.135, 0.13, 0.12, 0.1225, 0.115], [0.115, 0.13, 0.12, 0.1225, 0.1225, 0.13, 0.125, 0.135]]

        # List of N_Species sublists to store HMM models
        self.list_HMM = [[] for j in range(N_SPECIES)]

        # Create list with N_fish sublists to store observations sequence
        self.list_obs = [[] for j in range(N_FISH)]

        # Create list of already observed fish
        self.list_fish = []

        self.list_correct = [[] for j in range(N_SPECIES)]

    def guess(self, step, observations):
        """
        This method gets called on every iteration, providing observations.
        Here the player should process and store this information,
        and optionally make a guess by returning a tuple containing the fish index and the guess.
        :param step: iteration number
        :param observations: a list of N_FISH observations, encoded as integers
        :return: None or a tuple (fish_id, fish_type)
        """
        self.list_obs = split_observation(
            self.list_obs, observations)  # Separate observation for each fish

        current_fish = getFish(self.list_fish)  # Determine fish to be analysed

        if (step % N_FISH) == 20:   # Wait until step 20 and make a random prediction
            return (current_fish, random.randint(0, N_SPECIES-1))
        elif (step % N_FISH) > 20:
            probabilities = []
            # Go through all HMMs and determine the likelihood of the fish belonging to one of the stored species
            for i in range(N_SPECIES):
                suma = 0.0
                if self.list_HMM[i]:    # If the sub-list actually has trained HMMs

                    suma = (forward_descaled(self.list_HMM[i][-1][0], N_SPECIES, N_SPECIES, self.list_HMM[i][-1][1],
                                             N_SPECIES, N_EMISSIONS, self.list_HMM[i][-1][2], N_SPECIES, self.list_obs[current_fish], len(self.list_obs[current_fish])))

                    probabilities.append(suma)
                else:
                    probabilities.append(suma)  # Keep index order

            # HMM with most resemblence to the fish sequence
            return (current_fish, max(range(len(probabilities)), key=probabilities.__getitem__))
        else:
            return None

        # return (step % N_FISH, random.randint(0, N_SPECIES-1))

    def reveal(self, correct, fish_id, true_type):
        """
        This methods gets called whenever a guess was made.
        It informs the player about the guess result
        and reveals the correct type of that fish.
        :param correct: tells if the guess was correct
        :param fish_id: fish's index
        :param true_type: the correct type of the fish
        :return:
        """
        self.list_fish.append(
            fish_id)  # Store fish ID in list that controls which fishes have been analysed

        # Store fish id to correct index
        self.list_correct[true_type].append(fish_id)

        A_new = self.A
        B_new = self.B
        q_new = self.q
        obs = []

        for j in self.list_correct[true_type]:  # Append observation sequence
            for i in range(len(self.list_obs[j])):
                obs.append(self.list_obs[j][i])

        A_new, B_new, q_new = run_baum_welch(
            A_new, N_SPECIES, N_SPECIES, B_new, N_SPECIES, N_EMISSIONS, q_new, N_SPECIES, obs, len(obs), 20)    # Train HMM

        # Create array with A matrix, B matrix and q matrix
        mat = [A_new, B_new, q_new]

        self.list_HMM[true_type].append(mat)    # Assign to respective species
