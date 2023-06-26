import os

# Open "output_iterations.txt" file in data/3DSSG
f = open("../data/3DSSG/output_iterations.txt", "r")

# Read the file
lines = f.readlines()
set_l = set()
for l in lines:
    # Take the string after the first :
    l = l.split(":")[1]
    l = l.split()[0]

    # Add to set
    set_l.add(l)

# Print the set
print(set_l)