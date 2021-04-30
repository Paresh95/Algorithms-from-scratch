def linear_search(arr, x):
    """
    Finds the index position of an element in a list. 
    """

    for i, v in enumerate(arr):
        if v == x:
            return i
            break
    return -1 


if __name__ == "__main__":

    lst = [1, 3, 5, 63, 543, 3232, 43423]
    print(linear_search(lst, 3232))



