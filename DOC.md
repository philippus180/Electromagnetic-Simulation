Inspired by ThreeBlueOneBrown I want to create a Simulation, that visualises the electromagnetic fields caused by moving charges. To calculate the EM-fieds I need to implement some form of Maxwells Equations and solve them numerically by tiny changes in time. In the simulation your cursor should represent a charged particle, that you can move by moving the mouse. I imagine it to be very cool to see the influences of the movement on the field and especially the resulting EM-waves.

I will use python to calculate the physics. To visualize the fields I choose pygame to begin with, because I have some experience with it. Maybe later I look for a different graphics environment (maybe even matplotlib?).

In the last few weeks, I thougt a lot about this idea and I try to record as much of my thoughts in the development process as possible.

# First Steps, 21.01.2024
I create a simple pygame-loop with black background. For the behavior of the charges I define a class with charge, position, velocity and acceleration. The class has an update method, that uses Newtons law and the Lorentz-Force (MARK to future me: use relativistic mechanics). My goal now is to implement a charge, that follows the cursor. Then I will try to calculate its electric field. I think thats a big challenge, cause I dont exactly know how to handle a vectorfield in python.
Result for today: A circle at the position of the cursor and a lovely green charge that is oszillating in a constant magnetic field.

# Next Day, 22.01.2024


# Tuesday, 23.01.2024
I managed to add the magnitude of the electric field and visualize it by shaded color.

# Wednesday, 24.01.2024
By now, the E-field just contains the magnitude and does not point in any direction. Because of that, the other test charge only accelerates in one direction, no matter where the cursor is. To implement the electric field as a vector field, I set up an array of size n x m x 3. The screen size is n times m pixels, so every pixel gets assigned an E-vector. The coordinate system should have its origin in the middle of the screen. I create an array of the same shape as E, which contains the position vectors. In the process of rewriting the code i got really confused with two different coordinate systems. Not only do they have different dimension, but I also mixed them up when storing the positions of the charges. On top of that I did not understand all the different numpy arrays I created and made a lot of mistakes trying to do the right calculation with the arrays. Finally, I get most of the errors sorted and the electric field pushes the other charge away (if both are positively charged). To see what is going on, I add a grid of little arrows (by now just lines), that visualize the direction (and magnitude) of the electric field. That works out well and I see that there is still an issue with the electric field. The vectors sort of seem to be turned by 90 degree. A weird bug because the x and y axis were switched (at least sometimes). Todays result is quite delightful, you can see the electric field and push around the poor green charge. But still there are no electromagnetic waves.

# Thursday, 25.01.2024

# Saturday, 27.01.2024

# Sunday, 28.01.2024
Yesterday, the first waves appeared and the code seems to work. The simulation is quite slow, so I timed the key components of the simulation, that I thought would take the most time. Calculating the electric field, turning it into a color map and blit the colors onto the pygame screen. Blitting the pixels takes less than a microsecond and creating the colormap around 10 microseconds (on 300x300 px), so these are not a performance issue. But calculating the electric field takes a lot of time, depending heavily on the resolution of the field and also on the amount of saved charge positions. On a 300x300 pixel screen and the last 20 charge positions, the calculation takes around 300 μs, on 600x600 px even over a second. The challenge now is to speed up the calculations. Numpy uses just one core of CPU by now, so multiprocessing might be worth a try. Also I try to timed parts of the calculation  seperately to find potential weak parts.

