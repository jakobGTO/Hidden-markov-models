'''
    ------------------------------------------  Solution for HMM1 ------------------------------------------------------------- 
    
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


def forward(A, B, q, seq):
    '''
        Forward Algorithm for HMMs

        @param A(list(list(str))) : State Transition Probabilities
        @param B(list(list(str))) : Observation Probabilities
        @param q(list(list(str))) : Initial State Probabilities
        @param seq(list(list(str))) : Sequence of Observations

    '''
    A_row, A_col, A_data = split_line(A)
    # print('n_rows {}  n_cols = {}  data {} '.format(A_row, A_col, A_data))
    B_row, B_col, B_data = split_line(B)
    q_row, q_col, q_data = split_line(q)
    # B_row + 2 = no.states + start and end
    table = [[0 for j in range(B_col)] for j in range(int(seq[0]))]

    for i, f in enumerate(q_data[0]):   # Initialization
        # print("f {} b {} res {} ".format(float(f), float(B_data[i][int(seq[1])]), float(f)*float(B_data[i][int(seq[1])])))
        table[0][i] = float(f)*float(B_data[i][int(seq[1])])

    cur_seq = 2
    for i in range(1, int(seq[0])):
        last_row_table = table[i-1]
        table_aux = []
        pp = 0
        for x in range(A_row):
            suma = 0
            for y in range(A_col):
                #print("a {} b {}".format(last_row_table[y], A_data[y][x]))
                suma += float(last_row_table[y])*float(A_data[y][x])
            table_aux.append(suma)
        for p in range(B_row):
            pp = float(table_aux[p])*float(B_data[p][int(seq[cur_seq])])
            # print("a {} b {} res {}".format(table_aux[p], B_data[p][int(seq[cur_seq])], pp))
            table[i][p] = pp
        cur_seq = cur_seq + 1

    result = sum(table[-1])

    print(table)
    # print(table)


if __name__ == "__main__":

    A = input().split(" ")
    B = input().split(" ")
    q = input().split(" ")
    seq = input().split(" ")

    forward(A, B, q, seq)    # Forward Algorithm
