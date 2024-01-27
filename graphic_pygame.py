import pygame
import time
import numpy as np
from maxwell_calculation import Charge, Field_Area

# Initialize Pygame
pygame.init()

# Set up the screen
WIDTH, HEIGHT = (300, 300)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Maxwells Simulation")

# Set up colors
black = (0, 0, 0)
white = (255, 255, 255)
green = (21, 176, 26)



Space = Field_Area(WIDTH, HEIGHT, (-10, 10), (-10, 10))


mouse_charge = Space.add_charge(charge=-1, init_position = np.array([-5.,0,0]))
test_charge = Space.add_charge(charge=0.5, init_position = np.array([0,5.,0]))

electric_field_surface = pygame.Surface((WIDTH, HEIGHT))
electric_field_colors = np.ones((WIDTH, HEIGHT, 3), dtype=np.uint8)


print(electric_field_colors)
print(np.array([1,2,3]).shape)
print((np.array([[1.2],[5]]) @ np.array([1,0,3,5]).reshape(1,4)).round())
# electric_field_colors[200:300,200:455] =  np.linspace(0,127,255).reshape(255,1) @ np.array([2,1,0]).reshape(1,3)

# Space.set_test_e_field(mouse_charge.charge, (0,0,0))
electric_field_colors = Space.E_field_in_color(saturation_point=1)


down_pressed = False
up_pressed = False
right_pressed = False
left_pressed = False
space_pressed = False                
enter_pressed = False
backspace_pressed = False

i = 0
f = 0.01

# Main game loop
running = True
while running:
    i += 1
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_SPACE:
                space_pressed = True
            if event.key == pygame.K_DOWN:
                down_pressed = True
            if event.key == pygame.K_UP:
                up_pressed = True
            if event.key == pygame.K_RIGHT:
                right_pressed = True
            if event.key == pygame.K_LEFT:
                left_pressed = True
            if event.key == pygame.K_RETURN:
                enter_pressed = True
            if event.key == pygame.K_BACKSPACE:
                backspace_pressed = True

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                space_pressed = False
            if event.key == pygame.K_DOWN:
                down_pressed = False
            if event.key == pygame.K_UP:
                up_pressed = False
            if event.key == pygame.K_RIGHT:
                right_pressed = False
            if event.key == pygame.K_LEFT:
                left_pressed = False
            if event.key == pygame.K_RETURN:
                enter_pressed = False
            if event.key == pygame.K_BACKSPACE:
                backspace_pressed = False

    charge_movement = 0.2 * ((down_pressed  - up_pressed) * np.array([0,1,0])  + (right_pressed  - left_pressed) * np.array([1,0,0]) + (enter_pressed - backspace_pressed) * np.array([0,0,1])) 
    if (down_pressed  - up_pressed) and (right_pressed  - left_pressed):
        charge_movement /= np.sqrt(2)

    new_charge_pos = mouse_charge.position + charge_movement

    if space_pressed:
        new_charge_pos = np.array([np.cos(2*np.pi*f*i), np.sin(2*np.pi*f*i), 0]) * 5


    mouse_x, mouse_y = pygame.mouse.get_pos()

    ### UPDATE PHYSICS ###
    # mouse_charge.update(Space.dt)
    # mouse_charge.set_update(Space.dt, Space.position_at_index((mouse_x, mouse_y)))
    mouse_charge.set_update(Space.dt, new_charge_pos)
    test_charge.update(Space.dt)


    # Space.set_test_e_field(mouse_charge.charge, mouse_charge.position)
    if i%5 == 0:
        Space.calculate_e_field(mouse_charge)
        electric_field_colors = Space.E_field_in_color(saturation_point=0.5)

    pygame.surfarray.blit_array(electric_field_surface, electric_field_colors)

    # Fill the screen with black
    screen.fill(black)

    screen.blit(electric_field_surface, (0,0))

    pygame.draw.circle(screen, white, Space.index_of_position(mouse_charge.position), 2)
    pygame.draw.circle(screen, green, Space.index_of_position(test_charge.position), 2)
    # pygame.draw.circle(screen, white, Space.index_of_position(np.zeros(3)), 4)
    # pygame.draw.circle(screen, white, Space.index_of_position(np.array([1,0,0])), 3)
    # pygame.draw.circle(screen, white, Space.index_of_position(np.array([0,1,0])), 3)


    # for pos in Space.position[::10, ::10].reshape(-1, 3):
    #     start = Space.index_of_position(pos)
    #     end = start + Space.E_at_position(pos)[:2]
    #     pygame.draw.line(screen, white, start, end, width=1)
        

    # Update the display
    pygame.display.flip()



# Quit Pygame
pygame.quit()