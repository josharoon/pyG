import bezier
from matplotlib import pyplot as plt
# Create a new bezier.Curve object
curve = bezier.Curve()

#
curve.control_points = [(100, 100), (200, 200), (300, 300)]

# Plot the curve on a figure
fig, ax = plt.subplots()
ax.plot(curve)

# Show the figure
plt.show()