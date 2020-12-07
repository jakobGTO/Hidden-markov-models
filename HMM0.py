'''
    ------------------------------------------  Solution for HMM0 ------------------------------------------------------------- 
    
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


def dot_prod(x_col, y_row, data1, data2, ty):
    '''
        Dot product of two matrices

        @param x_col(int) : Number of columns
        @param y_row(int): Number of rows
        @param data1(list(list(str))) : Matrix A
        @param data2(list(list(str))) : Matrix B
        @param ty (int) : Control position of matrices on the dot product
    '''
    if ty == 0:  # A.B
        res = []
        for i in range(x_col):
            suma = 0
            for j in range(y_row):
                # print('a {}'.format(float(A_data[j][i])))
                suma = suma + (float(data1[0][j])*float(data2[j][i]))
            # print('s {}'.format(suma))
            res.append(suma)
        # print('res={}'.format(res))
        return res
    else:   # B.A
        res = []
        for i in range(x_col):
            suma = 0
            for j in range(y_row):
                # print('q {}'.format(float(q_data[0][j])))
                suma = suma + (float(data2[0][j])*float(data1[j][i]))
            # print('s {}'.format(suma))
            res.append(suma)
        # print('res={}'.format(res))
        return res


def problem_zero(A, B, q):
    '''
        Solution for HMM0

        @param A (list of str): State transition table
        @param B (list of str): State-observation table
        @param q (list of str): initial state probabilities
    '''

    A_row, A_col, A_data = split_line(A)
    B_row, B_col, B_data = split_line(B)
    q_row, q_col, q_data = split_line(q)

    # Display Data
    # print('n_rows {}  n_cols = {}  data {} '.format(A_row, A_col, A_data))
    # print('n_rows {}  n_cols = {}  data {} '.format(B_row, B_col, B_data))
    # print('n_rows {}  n_cols = {}  data {} '.format(q_row, q_col, q_data))

    qa = dot_prod(q_col, A_col, q_data, A_data, 0)

    aux = [qa]
    output = dot_prod(B_col, q_col, B_data, [qa], 1)
    o = [q_row, B_col]  # List to be printed
    for i in output:    # Join output values to o list
        o.append(i)

    print(' '.join(map(str, o)))    # Print result


if __name__ == "__main__":

    A = input().split(" ")
    B = input().split(" ")
    q = input().split(" ")

    problem_zero(A, B, q)
