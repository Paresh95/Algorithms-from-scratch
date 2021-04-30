def binary_search(arr, x):
    """
    Implimentation of the binary search algorithm to find the index 
    position of an element in a sorted list. 
    """

    low = 0 
    mid = 0
    high = len(arr) - 1

    while low <= high:
        mid = (high + low) // 2

        if arr[mid] < x:
            low = mid + 1

        elif arr[mid] > x:
            high = mid - 1

        else:
            return mid
        
    return -1

if __name__ == "__main__":

    lst = [1, 5, 6, 31, 65, 322, 645]
    print(binary_search(lst, 1))
