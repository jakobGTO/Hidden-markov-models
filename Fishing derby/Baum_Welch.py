'''
    ------------------------------------------  Baum-Welch Algorithm -------------------------------------------------------------

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


def forward(A, A_row, A_col, B, B_row, B_col, q, q_size, seq, seq_size):
    c = [0] * int(seq_size)

    alpha = [[0 for j in range(A_col)] for j in range(int(seq_size))]
    # compute alpha_0(i)

    for i in range(A_col):
        alpha[0][i] = float(q[i])*float(B[i][int(seq[0])])
        c[0] += alpha[0][i]

    # scale alpha_0(i)
    c[0] = 1/(c[0]+0.001)
    for i in range(A_row):
        alpha[0][i] *= c[0]

    # compute alpha_t(i)
    for t in range(1, int(seq_size)):
        c[t] = 0
        for i in range(A_col):
            alpha[t][i] = 0
            for j in range(A_col):
                alpha[t][i] += alpha[t-1][j]*float(A[j][i])

            alpha[t][i] = alpha[t][i] * float(B[i][int(seq[t])])
            c[t] += alpha[t][i]
        # scale alpha_t(i)
        # Add some noise to avoid zero division problems
        c[t] = 1/(c[t]+0.001)
        for i in range(A_col):
            alpha[t][i] = c[t]*alpha[t][i]
    return alpha, c


def forward_guess(A, A_row, A_col, B, B_row, B_col, q, q_size, seq, seq_size):
    c = [0] * int(seq_size)

    alpha = [[0 for j in range(A_col)] for j in range(int(seq_size))]
    # compute alpha_0(i)

    for i in range(A_col):
        alpha[0][i] = float(q[i])*float(B[i][int(seq[0])])
        c[0] += alpha[0][i]

    # scale alpha_0(i)
    c[0] = 1/(c[0]+0.001)
    for i in range(A_row):
        alpha[0][i] *= c[0]

    # compute alpha_t(i)
    for t in range(1, int(seq_size)):
        c[t] = 0
        for i in range(A_col):
            alpha[t][i] = 0
            for j in range(A_col):
                alpha[t][i] += alpha[t-1][j]*float(A[j][i])

            alpha[t][i] = alpha[t][i] * float(B[i][int(seq[t])])
            c[t] += alpha[t][i]
        # scale alpha_t(i)
        # Add some noise to avoid zero division problems
        c[t] = 1/(c[t]+0.001)
        for i in range(A_col):
            alpha[t][i] = c[t]*alpha[t][i]
    return sum(alpha[-1])


def forward_descaled(A, A_row, A_col, B, B_row, B_col, q, q_size, seq, seq_size):
    alpha = [[0 for j in range(A_col)] for j in range(int(seq_size))]
    # compute alpha_0(i)

    for i in range(A_col):
        alpha[0][i] = float(q[i])*float(B[i][int(seq[0])])

    # compute alpha_t(i)
    for t in range(1, int(seq_size)):
        for i in range(A_col):
            alpha[t][i] = 0
            for j in range(A_col):
                alpha[t][i] += alpha[t-1][j]*float(A[j][i])

            alpha[t][i] = alpha[t][i] * float(B[i][int(seq[t])])
    # print(alpha)
    # print(alpha[-1])
    #print("sum {}".format(sum(alpha[-1])))
    return sum(alpha[-1])


def backward(A, A_row, A_col, B, B_row, B_col, q, q_size, seq, seq_size, c):

    beta = [[0 for j in range(A_col)] for j in range(int(seq_size))]

    for i in range(A_col):
        beta[-1][i] = c[-1]

    for t in reversed(range(int(seq_size)-1)):
        for i in range(A_col):
            beta[t][i] = 0
            for j in range(A_col):
                beta[t][i] += float(A[i][j]) * float(B[j]
                                                     [int(seq[t+1])]) * beta[t+1][j]
            beta[t][i] *= c[t]
    return beta


def gamma_func(A, A_row, A_col, B, B_row, B_col, seq, seq_size, alpha, beta):

    gamma = [[0 for j in range(A_col)] for j in range(int(seq_size))]
    digamma = [[] for j in range(int(seq_size)-1)]
    for t in range(int(seq_size)-1):
        for i in range(A_col):
            gamma[t][i] = 0
            digamma[t].append([])
            for j in range(A_col):
                digamma[t][i].append(alpha[t][i] * float(A[i][j])
                                     * float(B[j][int(seq[t+1])]) * beta[t+1][j])
                gamma[t][i] += digamma[t][i][j]

    for i in range(A_col):
        gamma[-1][i] = alpha[-1][i]

    return digamma, gamma


def re_estimate(A, A_row, A_col, B, B_row, B_col, q, q_size, seq, seq_size, gamma, digamma, c):
    A_new = [[0 for j in range(A_col)] for j in range(A_col)]
    B_new = [[0 for j in range(B_col)] for j in range(A_col)]
    q_new = [0 for j in range(A_col)]
    # Restimate q
    for i in range(A_col):
        q_new[i] = gamma[0][i]

    # Restimate A

    for i in range(A_col):
        denom = 0
        for t in range(int(seq_size)-1):
            denom = denom + gamma[t][i]
        for j in range(A_col):
            numer = 0
            for t in range(int(seq_size)-1):
                numer = numer + digamma[t][i][j]
            A_new[i][j] = numer/(denom+0.001)

    # Restimate B
    for i in range(A_col):
        denom = 0
        for t in range(int(seq_size)):
            denom = denom + gamma[t][i]
        for j in range(B_col):
            numer = 0
            for t in range(0, int(seq_size)):
                # print(gamma[t][i])
                if(int(seq[t]) == j):
                    numer = numer + gamma[t][i]
            B_new[i][j] = numer/(denom+0.001)

    # Compute logProb
    logProb = 0
    for i in range(int(seq_size)):
        logProb = logProb + math.log(c[i])
    logProb = -logProb
    return logProb, A_new, B_new, q_new


def forward_first(A, A_row, A_col, B, B_row, B_col, q, q_size):
    probab = []
    # (len(q))
    # print(len(B))
    for i in range(B_row):
        suma = 0
        for j in range(B_col):
            suma += q[j]*B[j][i]
            #print("q {} b {}".format(q[j], B[j][i]))
        # print(suma)
        probab.append(suma)
    return max(range(len(probab)), key=probab.__getitem__)


def run_baum_welch(A, A_row, A_col, B, B_row, B_col, q, q_size, seq, seq_size, maxIters):
    oldLogProb = -(math.inf)

    for i in range(maxIters):
        # print(i)
        alpha, c = forward(A, A_row, A_col, B, B_row,
                           B_col, q, q_size, seq, seq_size)
        # print(alpha)
        beta = backward(A, A_row, A_col, B, B_row, B_col,
                        q, q_size, seq, seq_size, c)
        #print("A before di-gamma {}".format(A))
        digamma, gamma = gamma_func(
            A, A_row, A_col, B, B_row, B_col, seq, seq_size, alpha, beta)
        logProb, A_new, B_new, q_new = re_estimate(A, A_row, A_col, B, B_row,
                                                   B_col, q, q_size, seq, seq_size, gamma, digamma, c)
        # print(A)
        if logProb < oldLogProb:
            break
        else:
            oldLogProb = logProb
    #result(A, A_row, A_col, B, B_row, B_col)
    #print("A {} ".format(A))
    #print("B {} ".format(B))
    #print("q {} ".format(q))
    #print("c {} ".format(c))
    return A_new, B_new, q_new


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


if __name__ == "__main__":
    A = input().split(" ")
    B = input().split(" ")
    q = input().split(" ")
    seq = input().split(" ")

    A_row, A_col, A_data = split_line(A)
    B_row, B_col, B_data = split_line(B)
    q_row, q_col, q_data = split_line(q)

    oldLogProb = -(math.inf)
