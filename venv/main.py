import numpy as np

# Get Route Matrix
def get_Route_Matix(Matrix, Matrix_2):
    Route = np.array([[0 for i in range(len(Matrix)) ] for j in range(len(Matrix))])
    row = 0
    column = 0
    while row < len(Matrix):
        column = 0
        while column<len(Matrix):
            if Matrix[row][column] == 0:
                pass
            else:
                Route[row][column] = Matrix_2[row][column]
            column += 1
        row += 1
    return Route


# Find the Maximal complete subgraph for current Max_value
def get_maxclass_index(Matrix, partioned_index, value):
    index = 0

    result = [partioned_index[0][1]+1]

    # 判断是否partioned_index中的点都符合极大完全子图的要求
    while index < len(partioned_index):
        column = 0
        temp2 = 1
        temp = np.array(None)
        for column in range(index):
            temp2 = Matrix[partioned_index[index][0]][partioned_index[column][0]]
            if temp2 != 0:
                temp = np.c_[temp, np.array(temp2)]
                column += 1
            else:
                partioned_index.pop(index)
                break
        if temp2 != 0 :
            result += [partioned_index[index][0] + 1]
            index += 1
    index = 1

    # 排除当前极大子图
    partioned_index.insert(0,[partioned_index[0][1],partioned_index[0][1]])
    while index < len(partioned_index):
        for column in range(0, index):
            Matrix[partioned_index[index][0]][partioned_index[column][0]]= Matrix[partioned_index[index][0]][partioned_index[column][0]] - value
            Matrix[partioned_index[column][0]][partioned_index[index][0]] = Matrix[partioned_index[index][0]][partioned_index[column][0]]

        index += 1

    return result

# Get Max_value and its coordinate
def get_max_index(Matrix, former_max):
    max_value = 0
    max_value_index = []
    row = 0
    column = 0
    while column < len(Matrix[0]):
        row = column
        while row <len(Matrix):
            if Matrix[row][column] > max_value and former_max > Matrix[row][column]:
                max_value = Matrix[row][column]
                max_value_index = [row,column]
            row += 1
        column += 1
    return max_value, max_value_index

# determine that is that the Right Maximal complete subgraph?
def legal_max_value(Matrix, index):
    partitioned_matrix_index = []
    proper_value = 0
    row = 0
    while row < len(Matrix):
        if Matrix[row][index[1]] >= Matrix[index[0]][index[1]]:
            proper_value += 1
            partitioned_matrix_index.append([row,index[1]])
        row += 1

    if proper_value >= Matrix[index[0]][index[1]] + 1:
        return partitioned_matrix_index
    else :
        return None

# Determine Greatest Class Only containing TWO items
def get_two_couple(Matrix,Matrix_2, result):
    for row in range(len(Matrix)):
        for column in range(row + 1):
            if Matrix[row][column] == 1 and Matrix_2[row][column] == 0:
                result.append([column + 1, row + 1])
    return result

# Determine whether Matrix is Zero Matrix
def is_zero(Matrix):
    row = 0
    column = 0
    while row<len(Matrix):
        column = 0
        while column<len(Matrix[0]):
            if Matrix[row][column] != 0:
                return False
            column += 1
        row += 1
    return True

def main():

    '''Matrix = [
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 1, 1, 1],
        [0, 0, 1, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 1, 0],
        [0, 0, 1, 1, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 0]
    ]'''

    '''Matrix = [
        [0,1,1,0,0,1],
        [1,0,1,0,0,0],
        [1,1,0,0,1,1],
        [0,0,0,0,0,0],
        [0,0,1,0,0,1],
        [1,0,1,0,1,0],
    ]'''
    Matrix =[
        [0,1,1,1,0,0],
        [1,0,1,1,1,0],
        [1,1,0,1,0,1],
        [1,1,1,0,0,0],
        [0,1,0,0,0,1],
        [0,0,1,0,1,0],
    ]
    # Prepare data
    Matrix = np.array(Matrix)

    # Matrix * Matrix
    Matrix_2 = np.dot(Matrix, Matrix)

    # Result
    result_set = []

    # Determine Greatest Class Only containing TWO items
    get_two_couple(Matrix, Matrix_2, result_set)
    Route = get_Route_Matix(Matrix, Matrix_2)

    # While untill Route is Zero
    while not is_zero(Route):

        max_value, max_value_index = get_max_index(Route, Route.max()+1)
        partioned_matrix_index = legal_max_value(Route, max_value_index)

        # Determine whether max_value correlates with the Maximal complete subgraph，不符合，max_value = max_value -1 再次寻找
        while partioned_matrix_index is None:
            max_value, max_value_index = get_max_index(Route, max_value)
            partioned_matrix_index = legal_max_value(Route, max_value_index)

        result_set += [get_maxclass_index(Route, partioned_matrix_index, max_value)]

    # Add Single ones
    for i in range(1,len(Matrix)+1):
        for item in result_set:
            flag = 1
            if i in item:
                flag = 0
                break
        if flag:
            result_set.append([i])

    # Show results
    print(result_set)


main()
