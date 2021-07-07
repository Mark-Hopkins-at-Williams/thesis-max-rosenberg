import matplotlib.pyplot as plt


x = [0.785,0.794,0.766,0.727,0.78,0.792,0.743,0.705,0.85,0.839,0.594,0.702,0.788,0.668,0.585,0.651,0.728,0.683,0.714,0.604,0.505]
y = [99,99,99,99,99,499,499,599,799,1499,2799,3099,15999,22599,53799,82599,15099,103499,150099,35999,272399]
labs = ["society","environment","chemical reaction","chemical process","society","agency","area","situation","dog","canine","natural process","administrative unit","carnivore","nutriment","region","unit","bird","container","placental","vehicle","food"]
print("lens :", len(x),len(y),len(labs))
fig, ax = plt.subplots()

ax.scatter(x, y, alpha=0.7)

for i in [12,14,15,16,18,19,20]:
    ax.annotate(labs[i], (x[i],y[i]))

plt.xlabel("Similarity")
plt.ylabel("Epochs")
plt.show()