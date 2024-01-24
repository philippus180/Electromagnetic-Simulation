import pygame
import numpy as np
from maxwell_calculation import Charge, Field_Area

# Initialize Pygame
pygame.init()

# Set up the screen
WIDTH, HEIGHT = (600, 600)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Maxwells Simulation")

# Set up colors
black = (0, 0, 0)
white = (255, 255, 255)
green = (21, 176, 26)


# Set up physics
time_step = 0.1

mouse_charge = Charge(-10)
test_charge = Charge(1, init_position=np.array([300.,300.,0.]))

electric_field_surface = pygame.Surface((WIDTH, HEIGHT))
electric_field_colors = np.ones((WIDTH, HEIGHT, 3), dtype=np.uint8)

baum = Field_Area(WIDTH, HEIGHT, (-10, 10), (-10, 10))

print(electric_field_colors)
print(np.array([1,2,3]).shape)
print((np.array([[1.2],[5]]) @ np.array([1,0,3,5]).reshape(1,4)).round())
electric_field_colors[200:300,200:455] =  np.linspace(0,127,255).reshape(255,1) @ np.array([2,1,0]).reshape(1,3)

baum.set_test_e_field(mouse_charge.charge, mouse_charge.position)
electric_field_colors = baum.E_field_in_color(saturation_point=1)


# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    mouse_x, mouse_y = pygame.mouse.get_pos()

    ### UPDATE PHYSICS ###
    mouse_charge.set_update(time_step, np.array([mouse_x, mouse_y, 0]))
    test_charge.update(time_step,baum.E,1)

    
    baum.set_test_e_field(mouse_charge.charge, mouse_charge.position)
    electric_field_colors = baum.E_field_in_color(saturation_point=1)


    pygame.surfarray.blit_array(electric_field_surface, electric_field_colors)

    # Fill the screen with black
    screen.fill(black)

    screen.blit(electric_field_surface, (0,0))
    pygame.draw.circle(screen, white, (mouse_x, mouse_y), 5)
    pygame.draw.circle(screen, green, (test_charge.x(), test_charge.y()), 5)

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()