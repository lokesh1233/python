a, b, c = 15, 20, 10
print('summing all the varibales {}'.format(a+b+c))
print('summing all the varibales %d' (a+b+c))

num_list = range(100)
fiver_list = [num for num in num_list if num % 5 == 0]
print("=>".join(str(num) for num in fiver_list))

import matplotlib.pyplot as plt
plt.plot([1,2,3,4], [10,20,30,40], color='darkblue', linewidth=1, lable="line")
plt.scatter([0.6, 1.2, 3.4,2.2], [12,35,11,26], color="red", marker="o", label="Scatter")
plt.title("title for label")
plt.legend()
plt.show()