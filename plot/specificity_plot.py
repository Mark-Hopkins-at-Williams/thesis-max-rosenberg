import matplotlib.pyplot as plt


x = [34,37,54,133,34,238,140,112,281,353,464,583,678,820,712,1103,1481,1001,2155,819,2178]
y = [99,99,99,99,99,499,499,599,799,1499,2799,3099,15999,22599,53799,82599,15099,103499,150099,35999,272399]
labs = ["society","environment","chemical reaction","chemical process","society","agency","area","situation","dog","canine","natural process","administrative unit","carnivore","nutriment","region","unit","bird","container","placental","vehicle","food"]
print("lens :", len(x),len(y),len(labs))
fig, ax = plt.subplots()

ax.scatter(x, y, alpha=0.7)

for i in [12,14,15,16,18,19,20]:
    ax.annotate(labs[i], (x[i],y[i]))

plt.xlabel("Specificity")
plt.ylabel("Epochs")
plt.show()