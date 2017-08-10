def q(arr, down, up):
    print(arr)
    left = down
    right = up
    p = arr[(down + up) // 2]
    while left <= right:
        while p > arr[left]:
            left += 1
        while p < arr[right]:
            right -= 1
        if left <= right:
            temp = arr[left]
            arr[left] = arr[right]
            arr[right] = temp 
            left += 1
            right -= 1   
    if down <= right:
        q(arr, down, right)
    if left <= up:
        q(arr, left, up)