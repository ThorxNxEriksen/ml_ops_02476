from quickdraw import QuickDrawDataGroup
import matplotlib.pyplot as plt 

# Load a specific category (e.g., 'cat')
cats = QuickDrawDataGroup("cat")

# Iterate over the generator and show up to 5 drawings
for i, drawing in enumerate(cats.drawings):
    if i >= 5:
        break
    drawing.image.show()

plt.show()
