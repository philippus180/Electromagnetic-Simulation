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

# Create a clock object to control the frame rate
clock = pygame.time.Clock()

# Set the desired FPS
FPS = 6

# Set up colors
black = (0, 0, 0)
white = (255, 255, 255)
green = (21, 176, 26)

Space = Field_Area(WIDTH, HEIGHT,(-10, 10), (-10, 10))

mouse_charge = Space.add_charge(charge=-2)
# test_charge = Space.add_charge(charge=0.5, init_position = np.array([0,5.,0]))

electric_field_surface = pygame.Surface((WIDTH, HEIGHT))
electric_field_colors = np.ones((WIDTH, HEIGHT, 3), dtype=np.uint8)


draw_field_vectors = False
mouse_input = False

down_pressed = False
up_pressed = False
right_pressed = False
left_pressed = False
space_pressed = False                
enter_pressed = False
backspace_pressed = False

charge_speed = 0.1 # c

t = 0
f = 2 

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

            elif event.key == pygame.K_SPACE:
                space_pressed = True
            elif event.key == pygame.K_DOWN:
                down_pressed = True
            elif event.key == pygame.K_UP:
                up_pressed = True
            elif event.key == pygame.K_RIGHT:
                right_pressed = True
            elif event.key == pygame.K_LEFT:
                left_pressed = True
            elif event.key == pygame.K_RETURN:
                enter_pressed = True
            elif event.key == pygame.K_BACKSPACE:
                backspace_pressed = True

            elif event.key == pygame.K_0:
                charge_speed = Space.dx 
            elif event.key == pygame.K_1:
                charge_speed = Space.dx * 0.1
            elif event.key == pygame.K_2:
                charge_speed = Space.dx * 0.2
            elif event.key == pygame.K_3:
                charge_speed = Space.dx * 0.3
            elif event.key == pygame.K_4:
                charge_speed = Space.dx * 0.4
            elif event.key == pygame.K_5:
                charge_speed = Space.dx * 0.5
            elif event.key == pygame.K_6:
                charge_speed = Space.dx * 0.6
            elif event.key == pygame.K_7:
                charge_speed = Space.dx * 0.7
            elif event.key == pygame.K_8:
                charge_speed = Space.dx * 0.8
            elif event.key == pygame.K_9:
                charge_speed = Space.dx * 0.9

        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                space_pressed = False
            elif event.key == pygame.K_DOWN:
                down_pressed = False
            elif event.key == pygame.K_UP:
                up_pressed = False
            elif event.key == pygame.K_RIGHT:
                right_pressed = False
            elif event.key == pygame.K_LEFT:
                left_pressed = False
            elif event.key == pygame.K_RETURN:
                enter_pressed = False
            elif event.key == pygame.K_BACKSPACE:
                backspace_pressed = False


    mouse_x, mouse_y = pygame.mouse.get_pos()


    

    if space_pressed:
        new_charge_pos = np.array([np.cos(2*np.pi*f*t), np.sin(2*np.pi*f*t), 0]) * 0.5
        t += Space.dt
    elif 0:
        pass

    else:
        charge_movement = charge_speed * ((right_pressed - left_pressed) * np.array([1,0,0]) + (down_pressed - up_pressed) * np.array([0,1,0])  + (enter_pressed - backspace_pressed) * np.array([0,0,1])) 
        if (down_pressed  - up_pressed) and (right_pressed  - left_pressed):
            charge_movement /= np.sqrt(2)

        new_charge_pos = mouse_charge.position + charge_movement



    ### UPDATE PHYSICS ###

    if mouse_input:
        mouse_charge.set_update(Space.dt, Space.position_at_index((mouse_x, mouse_y)))
    else:
        mouse_charge.set_update(Space.dt, new_charge_pos)
        



    start = time.time()
    # Space.set_test_e_field(mouse_charge.charge, mouse_charge.position)
    Space.calculate_e_field(mouse_charge)
    electric_field_colors = Space.E_field_in_color(saturation_point=0.2)

    pygame.surfarray.blit_array(electric_field_surface, electric_field_colors)

    end = time.time()
    print(f'took {(end - start)*1000:.3f}')

    # Fill the screen with black
    screen.fill(black)
    screen.blit(electric_field_surface, (0,0))

    pygame.draw.circle(screen, white, Space.index_of_position(mouse_charge.position), 2)
    # pygame.draw.circle(screen, green, Space.index_of_position(test_charge.position), 2)
    # pygame.draw.circle(screen, white, Space.index_of_position(np.zeros(3)), 2)
    # pygame.draw.circle(screen, white, Space.index_of_position(np.zeros(3) + np.array([1,0,0])*Space.SPEED_OF_LIGHT*Space.dt), 2)
    # pygame.draw.circle(screen, white, Space.index_of_position(np.array([1,0,0])), 3)
    # pygame.draw.circle(screen, white, Space.index_of_position(np.array([0,1,0])), 3)



    start_time = time.time()


    if draw_field_vectors:
        for pos in Space.position[::10, ::10].reshape(-1, 3):
            start = Space.index_of_position(pos)
            end = Space.index_of_position(pos[:2] + Space.E_at_position(pos)[:2] / max(0.5, 2*np.linalg.norm(Space.E_at_position(pos)[:2])))
            pygame.draw.line(screen, white, start, end, width=1)
        
    end_time = time.time()
    print(f'took {(end_time - start_time)*1000:.3f}')


    # Update the display
    pygame.display.flip()


    # Cap the frame rate to the desired FPS
    clock.tick(FPS)


# Quit Pygame
pygame.quit()