def binary_search(arr, x, low, high):
    mid = (high + low) // 2
    if arr[mid] == x:
        print(mid)
    elif arr[mid] > x:
        left = arr[:mid]
        return binary_search(arr, x, 0, mid)
    elif x > arr[mid]:
        right = arr[mid+1:]
        binary_search(arr, x, mid+1, high)
