# Electromagnetic Simulation
Inspired by ThreeBlueOneBrown ([link to video](https://www.youtube.com/watch?v=aXRTczANuIs)) I wanted to create a simulation, that visualises the electromagnetic fields caused by moving charges. To calculate the EM-fieds I need to implement some form of Maxwells Equations and solve them numerically by tiny changes in time.
In the simulation your cursor represents a charged particle, that you can move by moving the mouse. It is very nice to see the influences of the movement on the field and especially the resulting EM-waves.


# Requirements
You need the following python packages to run the simulation: pygame, numpy, numexpr, cv2. These packages can be installed by running the following commands in a terminal:
```bash
pip install pygame
pip install numpy
pip install numexpr
pip install opencv-python
```
To start the simulation please run the file 'graphics_pygame.py':
```bash
python graphics_pygame.py
```

# Features
The little dot represents a charge. It causes an electromagnetic field around it that is shown in blue color. The brightness of the color represents the magnitude of the electric field. The brighter the color, the stronger the electric field. The screen only shows the x-y-plane, but the field is calculated as 3d-vector. To see the direction of the field press v and a grid of little lines gets drawn on the screen. Every line points in the direction of the x- and y-components of the electric field at that point. Drawing the vectors slows down the simulation.

One can move around the charge using the arrow keys and suddenly there are electromagnetic waves appearing when the charge accelerates or slows down. After pressing the space bar, the charge will follow the position of the mouse cursor. This will lead to weird behavior of the field, because the cursor can move around faster than the speed of light. But it looks nice.
To enjoy and study the EM-waves there are some exemplary motions implemented. First up is simple harmonic motion along each axis. Then there is a circular movement, where the charge orbits around the origin. Feel free to change the frequency and watch the effects on the waves.

|Key|Action|
|:---:|:---:|
|v|toggle vectorfield|
|space|toggle mouse input|
|arrow keys or wasd|move in x/y-direction|
|enter / backspace or q / e|move in z-direction|
|1 ... 9|set movement speed to 0.x of c|
|0|set movement speed equal to c|
|x / y / z|oszillate charge in x/y/z-direction|
|c|circle charge around origin|
|f / g|increase / decrease frequency of oszillation|
|r / t|increase / decrease amplitude of oszillation|
|mouse button left / right|increase / decrease charge|


# Implementation
I use python (especially numpy) to calculate the physics. To visualize the fields I choose pygame.
