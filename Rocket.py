import math


gravity = 9.81
mass = 1 # kg
angleOffset = 0 # degrees

data = []

with open("AeroTech_HP-I500T.eng", "r") as file:
    for line in file:
        line = line.strip()

        # Ignore blank lines and comments
        if not line or line.startswith(";"):
            continue

        parts = line.split()

        # Data rows contain two numbers
        if len(parts) == 2:
            try:
                time = float(parts[0])
                thrust = float(parts[1])
                data.append((time, thrust))
            except ValueError:
                pass

print(data)


acceleration = []
for i in range(len(data)):
    acceleration.append((data[i][0], data[i][1]/mass ))

print(acceleration)

def getPosition():
    heightV = 0
    height = 0
    xV = 0
    xOffset = 0

    
    for i in range(len(acceleration)-1):
        dx = acceleration[i+1][0] - acceleration[i][0]
        heightV += (acceleration[i][1]*math.cos(math.radians(angleOffset))-gravity)*dx
        xV += acceleration[i][1]*math.sin(math.radians(angleOffset))*dx
        height += heightV * dx
        xOffset += xV * dx

    print(heightV)
    height += 0.5*heightV*heightV/gravity
    xOffset += heightV/gravity*xV
    print(heightV/gravity)

    return (height, xOffset)

def getTime():
    v = 0
    
    for i in range(len(acceleration)-1):
        dx = acceleration[i+1][0] - acceleration[i][0]
        v += acceleration[i][1]*math.cos(math.radians(angleOffset))*dx

    return acceleration[-1][0] + v/gravity

print("Position: ", getPosition())
print("Time: ", getTime())
