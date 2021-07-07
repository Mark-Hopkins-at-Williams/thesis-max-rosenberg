import matplotlib.pyplot as plt

x = [0.25,0.33,0.53,0.47,0.25,0.57,0.35,0.31,0.95,0.31,0.22,0.52,0.02,0.03,0.04,0.042,0.029,0.068,0.025,0.15,0.0098]
y = [99,99,99,99,99,499,499,599,799,1499,2799,3099,15999,22599,53799,82599,15099,103499,150099,35999,272399]
labs = ["society","environment","chemical reaction","chemical process","society","agency","area","situation","dog","canine","natural process","administrative unit","carnivore","nutriment","region","unit","bird","container","placental","vehicle","food"]
print("lens :", len(x),len(y),len(labs))
fig, ax = plt.subplots()

ax.scatter(x, y, alpha=0.6)

for i in [14,15,16,18,19,20]:
    ax.annotate(labs[i], (x[i],y[i]))

plt.xlabel("Flatness")
plt.ylabel("Epochs")
plt.show()