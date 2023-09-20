from datetime import date

class MyClass:
    def __init__(self):
        print("hello world")
    def print(self):
        print("hello world!!!")
    def __str__(self):
        return "hello\n\nhello"

cl = MyClass()
print(cl)
print([cl])
print([])
print([1, 2, 3])
print([1, 2, 3])
print(",".join(list(map(lambda x : str(x), [cl, cl]))))
print(date(2000, 10, 10))
print(None)
arr = [1]
print(arr[1])