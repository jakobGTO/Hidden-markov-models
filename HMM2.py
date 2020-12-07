'''
    ------------------------------------------  Solution for HMM2 -------------------------------------------------------------

                                        KTH Royal Institute of Technology
                                            M.Sc Machine Learning 20/21

                                        DD280 - Artificial Intelligence

                                        Diogo Pinheiro & Jakob Lindén

    -------------------------------------------------------------------------------------------------------------------------
'''


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

    viterbi(A, B, q, seq)    # Forward Algorithm
