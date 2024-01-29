import time

import numpy as np
import pygame

from maxwell_calculation import Field_Area


WIDTH, HEIGHT = (300, 300)
FPS = 6

dt = 0.05
speed_of_light = 10

frequency = 20
amplitude = 0.2

Space = Field_Area(WIDTH, HEIGHT,(-10, 10), (-10, 10), dt, speed_of_light)

mouse_charge = Space.add_charge(charge=1)
# test_charge = Space.add_charge(charge=20, init_position = np.array([1,7.,0]))

# Set up colors
black = (0, 0, 0)
white = (255, 255, 255)
green = (21, 176, 26)
red = (255, 0, 0)
blue = (0, 0, 255)


# Initialize Pygame
pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Maxwells Simulation")
clock = pygame.time.Clock()

electric_field_surface = pygame.Surface((WIDTH, HEIGHT))


draw_field_vectors = False
mouse_input = False

down_pressed = False
up_pressed = False
right_pressed = False
left_pressed = False           
enter_pressed = False
backspace_pressed = False

x_pressed = False
y_pressed = False
z_pressed = False
c_pressed = False

charge_speed = 0.1 # c

t = 0
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
                mouse_input = not mouse_input
            elif event.key == pygame.K_v:
                draw_field_vectors = not draw_field_vectors
            elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                down_pressed = True
            elif event.key == pygame.K_UP or event.key == pygame.K_w:
                up_pressed = True
            elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                right_pressed = True
            elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                left_pressed = True
            elif event.key == pygame.K_RETURN or event.key == pygame.K_e:
                enter_pressed = True
            elif event.key == pygame.K_BACKSPACE or event.key == pygame.K_q:
                backspace_pressed = True
            elif event.key == pygame.K_x:
                x_pressed = True
            elif event.key == pygame.K_y:
                y_pressed = True
            elif event.key == pygame.K_z:
                z_pressed = True
            elif event.key == pygame.K_c:
                c_pressed = True

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


            elif event.key == pygame.K_f:
                frequency += 4
            elif event.key == pygame.K_g:
                frequency -= 4
            elif event.key == pygame.K_r:
                amplitude += 0.05
            elif event.key == pygame.K_t:
                amplitude -= 0.05


        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_DOWN or event.key == pygame.K_s:
                down_pressed = False
            elif event.key == pygame.K_UP or event.key == pygame.K_w:
                up_pressed = False
            elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                right_pressed = False
            elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                left_pressed = False
            elif event.key == pygame.K_RETURN or event.key == pygame.K_q:
                enter_pressed = False
            elif event.key == pygame.K_BACKSPACE or event.key == pygame.K_e:
                backspace_pressed = False

            elif event.key == pygame.K_x:
                x_pressed = False
            elif event.key == pygame.K_y:
                y_pressed = False
            elif event.key == pygame.K_z:
                z_pressed = False
            elif event.key == pygame.K_c:
                c_pressed = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_charge.charge += 0.5
            elif event.button == 3:
                mouse_charge.charge -= 0.5

    mouse_x, mouse_y = pygame.mouse.get_pos()


    if c_pressed:
        new_charge_pos = np.array([np.cos(frequency*t), np.sin(frequency*t), 0]) * amplitude
        t += Space.dt
    elif x_pressed:
        new_charge_pos = np.array([np.sin(frequency*t), 0, 0]) * amplitude
        t += Space.dt
    elif y_pressed:
        new_charge_pos = np.array([0, np.sin(frequency*t), 0]) * amplitude
        t += Space.dt
    elif z_pressed:
        new_charge_pos = np.array([0, 0, np.sin(frequency*t)]) * amplitude
        t += Space.dt
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

    # test_charge.update(Space.dt)
        

    start_time = time.time()
    # Space.set_test_e_field(mouse_charge.charge, mouse_charge.position)
    Space.calculate_e_field(mouse_charge)
    electric_field_colors = Space.E_field_in_color(saturation_point=0.5)

    pygame.surfarray.blit_array(electric_field_surface, electric_field_colors)

    end_time = time.time()
    print(f'Calculate E-field took {(end_time - start_time)*1000:.3g} μs')

    # Fill the screen with black
    screen.fill(black)
    screen.blit(electric_field_surface, (0,0))

    if mouse_charge.charge == 0:
        pygame.draw.circle(screen, white, Space.index_of_position(mouse_charge.position), 2)
    elif mouse_charge.charge > 0:
        pygame.draw.circle(screen, red, Space.index_of_position(mouse_charge.position), 2 * np.ceil(mouse_charge.charge))
    elif mouse_charge.charge < 0:
        pygame.draw.circle(screen, blue, Space.index_of_position(mouse_charge.position), 2 * np.ceil(-mouse_charge.charge))
    # pygame.draw.circle(screen, green, Space.index_of_position(test_charge.position), 2)
    

    if draw_field_vectors:
        start_time = time.time()
        for pos in Space.position[::10, ::10].reshape(-1, 3):
            start = Space.index_of_position(pos)
            end = Space.index_of_position(pos[:2] + Space.E_at_position(pos)[:2] / max(0.25, 2*np.linalg.norm(Space.E_at_position(pos)[:2])))
            pygame.draw.line(screen, white, start, end, width=1)
            
        end_time = time.time()
        print(f'Drawing vectorfield took {(end_time - start_time)*1000:.3g} μs')


    # Update the display
    pygame.display.flip()


    # Cap the frame rate to the desired FPS
    clock.tick(FPS)


# Quit Pygame
pygame.quit()