import math
import pygame

pygame.init()

window = pygame.display.set_mode((800, 600))

gravity = 9.81
mass = 1.759 # kg
angleOffset = 0 # degrees

airDensity = 1.2 #Air density (kg/m^3)
dragCoefficient = 0.75 #Drag coefficient (dimensionless)
area = 0.0026238896 #Cross-sectional area (m^2)
dt = 0.0001

#graph variables
scaleTime = 15
scaleY = 1/2
Yoffset = 50
# yellow position
#red Speed
#white acceleration

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

#print(data)


acceleration = []
for i in range(len(data)):
    acceleration.append((data[i][0], data[i][1]/mass ))

#print(acceleration)

 

def getPosition():
    heightV = 0
    height = 0
    xV = 0
    xOffset = 0
    time = 0

    angle = angleOffset/180 * 3.14159265

    apedogee = False

    i = 0

    pygame.draw.line(window, (0, 0, 255), (0, Yoffset), (800, Yoffset), 1)
    
    while height >= 0:
        
                

        speed = (heightV**2 + xV**2)**(1/2)

        k = 0.5 * airDensity * dragCoefficient * area

        dragX = -k * speed * xV
        dragY = -k * speed * heightV

        # Calculate acceleration
        if i < len(acceleration):
            accelX = acceleration[i][1] * math.sin(angle) + dragX / mass
            accelY = acceleration[i][1] * math.cos(angle) - gravity + dragY / mass
        else:
            accelX = dragX / mass
            accelY = -gravity + dragY / mass

        # Update position using current velocity
        height += heightV * dt + 0.5 * accelY * dt**2
        xOffset += xV * dt + 0.5 * accelX * dt**2

        # Update velocity
        heightV += accelY * dt
        xV += accelX * dt

        # graph
        pygame.draw.rect(window, (255, 255, 0), (time*scaleTime, height*scaleY+Yoffset, 1, 1))
        pygame.draw.rect(window, (255, 0, 0), (time*scaleTime, heightV*scaleY+Yoffset, 1, 1))
        pygame.draw.rect(window, (255, 255, 255), (time*scaleTime, accelY*scaleY+Yoffset, 1, 1))
        
        time += dt
        if i < len(acceleration) and acceleration[i][0] < time:
            i += 1

        if (not apedogee) and heightV < 0:
            apedogee = True
            print("Apogee reached at time: ", time, " seconds")
            print("Apogee height: ", height, " meters")
            print("Apogee xOffset: ", xOffset, " meters")
        

    print("Rocket has landed at time: ", time, " seconds")

getPosition()

pygame.display.flip()

# Keep the window open
while pygame.event.wait().type != pygame.QUIT:
    pass

pygame.quit()