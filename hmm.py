'''
    ------------------------------------------  Solution for HMM3 -------------------------------------------------------------

                                        KTH Royal Institute of Technology
                                            M.Sc Machine Learning 20/21

                                        DD280 - Artificial Intelligence

                                        Diogo Pinheiro & Jakob Lindén

    -------------------------------------------------------------------------------------------------------------------------
'''

import math


def split_line(line):
    '''
        Split List of str into no.rows, no.columns and data, according to the input data provided

        @param line(List(str)) : Line read from input file
    '''
    n_rows = (int)(line[0])
    n_col = (int)(line[1])
    data = line[2:]

    list = []
    mul = 0
    for i in range(n_rows):
        list_aux = []

        for j in range(mul, mul+n_col):
            list_aux.append(data[j])
        mul = j+1
        list.append(list_aux)
    return n_rows, n_col, list


def forward(A, B, q, seq):
    c = [0] * int(seq[0])

    alpha = [[0 for j in range(A_col)] for j in range(int(seq[0]))]

    # compute alpha_0(i)
    for i in range(A_col):
        alpha[0][i] = float(q_data[0][i])*float(B_data[i][int(seq[1])])
        c[0] += alpha[0][i]

    # scale alpha_0(i)
    c[0] = 1/c[0]
    for i in range(A_row):
        alpha[0][i] *= c[0]

    # compute alpha_t(i)
    for t in range(1, int(seq[0])):
        c[t] = 0
        for i in range(A_col):
            alpha[t][i] = 0
            for j in range(A_col):
                alpha[t][i] += alpha[t-1][j]*float(A_data[j][i])

            alpha[t][i] = alpha[t][i] * float(B_data[i][int(seq[t+1])])
            c[t] += alpha[t][i]
        # scale alpha_t(i)
        c[t] = 1/c[t]
        for i in range(A_col):
            alpha[t][i] = c[t]*alpha[t][i]
    return alpha, c


def backward(A, B, q, seq, c):
    seq_new = seq[1:]
    beta = [[0 for j in range(A_col)] for j in range(int(seq[0]))]

    #Let beta_T-1(i) = 1, scaled by C_T-1
    for i in range(A_col):
        beta[-1][i] = c[-1]

    #Apply beta pass
    for t in reversed(range(int(seq[0])-1)):
        for i in range(A_col):
            beta[t][i] = 0
            for j in range(A_col):
                beta[t][i] += float(A_data[i][j]) * float(B_data[j]
                                                          [int(seq_new[t+1])]) * beta[t+1][j]
            # scale beta_t(i) with same scale factor as alpha_t(i)
            beta[t][i] *= c[t]
    return beta


def gamma_func(A, B, seq, alpha, beta):

    seq_new = seq[1:]

    gamma = [[0 for j in range(A_col)] for j in range(int(seq[0]))]
    digamma = [[] for j in range(int(seq[0])-1)]

    # using scaled alpha and beta, no need to normalize digamma_t(i,j)
    for t in range(int(seq[0])-1):
        for i in range(A_col):
            gamma[t][i] = 0
            digamma[t].append([])
            for j in range(A_col):
                digamma[t][i].append(alpha[t][i] * float(A_data[i][j])
                                     * float(B_data[j][int(seq_new[t+1])]) * beta[t+1][j])
                gamma[t][i] += digamma[t][i][j]

    # special case for gamma_T-1(i), append last scaled alpha as last gamma
    for i in range(A_col):
        gamma[-1][i] = alpha[-1][i]

    return digamma, gamma


def re_estimate(A, B, q, seq, gamma, digamma, c):
    # Restimate q
    for i in range(A_col):
        q_data[0][i] = gamma[0][i]

    # Restimate A
    for i in range(A_col):
        denom = 0
        for t in range(int(seq[0])-1):
            denom = denom + gamma[t][i]
        for j in range(A_col):
            numer = 0
            for t in range(int(seq[0])-1):
                numer = numer + digamma[t][i][j]
            A_data[i][j] = numer/(denom + 0.001)

    # Restimate B
    for i in range(A_col):
        denom = 0
        for t in range(int(seq[0])):
            denom = denom + gamma[t][i]
        for j in range(B_col):
            numer = 0
            for t in range(0, int(seq[0])):
                # print(gamma[t][i])
                if(int(seq[t+1]) == j):
                    numer = numer + gamma[t][i]
            B_data[i][j] = numer/(denom + 0.001)

    # Compute log[P(O|lambda)]
    logProb = 0
    for i in range(int(seq[0])):
        logProb = logProb + math.log(c[i])
    logProb = -logProb
    return logProb


def run_baum_welch(maxIters, oldLogProb):
    # start logprob with -inf
    oldLogProb = -(math.inf)
    for i in range(maxIters):
        # get current alpha,betta,gamma and digamma tables, reestimate them with func re_estimate
        # stop if current iteration logProbability is lower than the old one or if maxIterations is reached.
        alpha, c = forward(A, B, q, seq)
        beta = backward(A, B, q, seq, c)
        digamma, gamma = gamma_func(A, B, seq, alpha, beta)
        logProb = re_estimate(A, B, q, seq, gamma, digamma, c)
        if logProb < oldLogProb:
            break
        else:
            oldLogProb = logProb
    # call result function used to print result matrices
    result(A_data, A_row, A_col, B_data, B_row, B_col)


def result(A, A_row, A_col, B, B_row, B_col):
    A_output = [A_row, A_col]
    for i in range(A_row):
        for j in range(A_col):
            A_output.append(A[i][j])

    B_output = [B_row, B_col]
    for i in range(B_row):
        for j in range(B_col):
            B_output.append(B[i][j])

    print(' '.join(map(str, A_output)))
    print(' '.join(map(str, B_output)))


def viterbi(A, B, q, seq):
    '''
        Viterbi Algorithm for HMMs

        @param A(list(list(str))) : State Transition Probabilities
        @param B(list(list(str))) : Observation Probabilities
        @param q(list(list(str))) : Initial State Probabilities
        @param seq(list(list(str))) : Sequence of Observations

    '''

    A_row, A_col, A_data = split_line(A)
    # print('n_rows {}  n_cols = {}  data {} '.format(A_row, A_col, A_data))
    B_row, B_col, B_data = split_line(B)
    q_row, q_col, q_data = split_line(q)

    table = [[0 for j in range(B_col)] for j in range(int(seq[0]))]
    state_seq_table = []  # Store argmax state for each observation

    for i, f in enumerate(q_data[0]):   # Initialization
        # print("f {} b {} res {} ".format(float(f), float(B_data[i][int(seq[1])]), float(f)*float(B_data[i][int(seq[1])])))
        table[0][i] = float(f)*float(B_data[i][int(seq[1])])

    cur_seq = 2         # Current observation
    argmax_state = []   # Store argmax state for each observation

    tab_line_state = []

    for i in range(1, int(seq[0])):
        last_row_table = table[i-1]
        max_prob = []       # Store max values for each observation

        tab_aux_line = []   # Store max probability table for all states in each observation
        tab_aux_line_state = []

        for x in range(A_row):
            tab_aux = []
            for y in range(A_col):
                # print("a {} b {} c {} res {}".format(
                #    last_row_table[y], A_data[y][x], B_data[x][int(seq[cur_seq])], float(last_row_table[y])*float(A_data[y][x])*float(B_data[x][int(seq[cur_seq])])))
                tab_aux.append(
                    float(last_row_table[y])*float(A_data[y][x])*float(B_data[x][int(seq[cur_seq])]))
            tab_aux_line.append(max(tab_aux))
            tab_aux_line_state.append(
                max(range(len(tab_aux)), key=tab_aux.__getitem__))
        tab_line_state.append(tab_aux_line_state)

        
        table[i] = tab_aux_line    # Append
        print(table)
        max_value = max(tab_aux_line)    # Max value in list
        # Index of max value in list
        max_index = max(range(len(tab_aux_line)), key=tab_aux_line.__getitem__)
        max_prob.append(max_value)

        # print('Max Value {} ; Argmax State {}'.format(
        #    max_value, tab_aux_line_state[max_index]))
        argmax_state.append(max_index)
        state_seq_table.append(tab_aux_line_state[max_index])
        cur_seq = cur_seq + 1   # Next observations

    # Backtrace
    print(argmax_state)
    print(state_seq_table)
    backtrace = [argmax_state[-1],
                 state_seq_table[-1]]   # Create in reverse order
    # Get all states, except the last two (already listed)
    for i in range(int(seq[0])-3, -1, -1):
        backtrace.append(tab_line_state[i][state_seq_table[i+1]])

    backtrace.reverse()  # Reverse List

    print(' '.join(map(str, backtrace)))    # Print result

if __name__ == "__main__":
    A = input().split(" ")
    B = input().split(" ")
    q = input().split(" ")
    seq = input().split(" ")

    A_row, A_col, A_data = split_line(A)
    B_row, B_col, B_data = split_line(B)
    q_row, q_col, q_data = split_line(q)

    alpha, c = forward(A, B, q, seq)
    beta = backward(A, B, q, seq, c)
    digamma, gamma = gamma_func(A, B, seq, alpha, beta)

    maxIters = 100000
    oldLogProb = -(math.inf)

    run_baum_welch(maxIters, oldLogProb)
