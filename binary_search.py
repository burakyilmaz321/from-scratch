def binary_search(arr, l, r), x):
    m = len(arr)
    if x == arr[m//2]:
        print(m//2)
    elif x < arr[m//2]:
        left = arr[:m//2]
        binary_search(left, x)
    elif x > arr[m//2]:
        right = arr[m//2+1:]
        binary_search(right, x)
