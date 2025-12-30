# day_3.py
data = [10, 20, 30, 40, 50]

# Function to compute sum
def sum_list(lst):
    total = 0
    for x in lst:
        total += x
    return total

# Function to compute average
def avg_list(lst):
    return sum_list(lst) / len(lst)

# Function to find max
def max_list(lst):
    max_val = lst[0]
    for x in lst:
        if x > max_val:
            max_val = x
    return max_val

print("Sum:", sum_list(data))
print("Average:", avg_list(data))
print("Max:", max_list(data))