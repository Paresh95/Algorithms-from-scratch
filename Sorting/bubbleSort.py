def bubble_sort(arr):
    """
    Uses the bubble sort algorithm to sort a list
    """

    n = len(arr)

    for i in range(n-1):
        for j in range(n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]

    return arr 


if __name__ == "__main__":

    lst = [1, 2, 2, 4, 1, 656, 12]
    print(bubble_sort(lst))