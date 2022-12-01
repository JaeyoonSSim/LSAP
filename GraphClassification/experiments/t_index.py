def sol(): 
    a = list(map(float, input().split())) 
    aidx = []

    for i, ele in enumerate(a):
        aidx.append([ele, i + 1])
    print("Result: ")
    
    return sorted(aidx, key=lambda k: k[0])


if __name__ == "__main__":
    print(sol()) # Input : values -> Output : values and indices in ascending order