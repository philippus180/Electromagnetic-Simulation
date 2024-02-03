import time

import cv2
import numexpr as ne
import numpy as np
import scipy


def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} took {execution_time*1000:.4g} ms to execute.")
        return result
    return wrapper


BLUE = (0,0,255)
RED = (255,0,0)


def add_vector_to_list(list, vector):
    return np.concatenate((list, vector[np.newaxis,:]))

class Charge():
    max_length_old = 40

    def __init__(self, field: 'Field_Area', charge=1, mass=1, init_position=np.zeros(3), init_velocity=np.zeros(3)) -> None:
        self.charge = charge
        self.mass = mass
        self.position = init_position # np.array([x,y,z])
        self.velocity = init_velocity
        self.acceleration = np.zeros(3)
        self.field = field

        self.old_positions = np.tile(init_position, (self.max_length_old,1))
        self.old_velocities = np.tile(init_velocity, (self.max_length_old,1))
        self.old_accelerations = np.tile(self.acceleration, (self.max_length_old,1))

        self.light_travel_distance = np.arange(self.max_length_old, 0, -1) * field.speed_of_light * field.dt

    def lorentz_force(self):
        return self.charge * (self.field.E_at_position(self.position) )# + np.cross(self.velocity, self.field.B(self.position)))
    
    def update(self, dt):
        self.old_positions = add_vector_to_list(self.old_positions, self.position)
        self.old_velocities = add_vector_to_list(self.old_velocities, self.velocity)
        self.old_accelerations = add_vector_to_list(self.old_accelerations, self.acceleration)
        if len(self.old_positions) > self.max_length_old:
            self.old_positions = np.delete(self.old_positions, 0, axis=0)
            self.old_velocities = np.delete(self.old_velocities, 0, axis=0)
            self.old_accelerations = np.delete(self.old_accelerations, 0, axis=0)

        self.acceleration = self.lorentz_force() / self.mass
        self.position += self.velocity * dt
        self.velocity += self.acceleration * dt

    def set_update(self, dt, pos):
        self.old_positions = add_vector_to_list(self.old_positions, self.position)
        self.old_velocities = add_vector_to_list(self.old_velocities, self.velocity)
        self.old_accelerations = add_vector_to_list(self.old_accelerations, self.acceleration)
        if self.old_positions.shape[0] > self.max_length_old:
            self.old_positions = np.delete(self.old_positions, 0, axis=0)
            self.old_velocities = np.delete(self.old_velocities, 0, axis=0)
            self.old_accelerations = np.delete(self.old_accelerations, 0, axis=0)
        old_pos = self.position
        old_vel = self.velocity
        self.position = pos
        self.velocity = (self.position - old_pos) / dt
        self.acceleration = (self.velocity - old_vel) / dt



class Field_Area():
    def __init__(self, x_resolution, y_resolution, x_range, y_range, dt, c) -> None:
        self.E = np.zeros((x_resolution, y_resolution, 3))
        self.E_norm = np.zeros((x_resolution, y_resolution))
        self.B = np.zeros((x_resolution, y_resolution, 3))

        self.x,self.y = np.indices((x_resolution, y_resolution))

        self.speed_of_light = c
        self.dt = dt
        self.dx = self.dt * self.speed_of_light

        self.x_resolution = x_resolution
        self.y_resolution = y_resolution

        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range

        self.x_width = self.x_max - self.x_min
        self.y_width = self.y_max - self.y_min

        self.pixel_size = (self.x_width / (x_resolution - 1), self.y_width / (y_resolution - 1))

        x = np.linspace(self.x_min, self.x_max, x_resolution)
        y = np.linspace(self.y_min, self.y_max, y_resolution)
        x_grid = np.tile(x, (y_resolution, 1))
        y_grid = np.tile(y, (x_resolution, 1)).T
        z_grid = np.zeros_like(x_grid)

        self.position = np.stack((y_grid, x_grid, z_grid), axis=-1)

        self.color = np.array((40, 150, 240), dtype=np.uint8)
        self.color_field = np.tile(self.color, (self.x_resolution, self.y_resolution, 1))

        self.charges = []

    def add_charge(self, **charge_properties):
        new_charge = Charge(self, **charge_properties)
        self.charges.append(new_charge)
        return new_charge

    def set_color(self, rgb_color):
        self.color = np.array(rgb_color, dtype=np.uint8)
        self.color_field = np.tile(self.color, (self.x_resolution, self.y_resolution, 1))
        return


    def position_at_index(self, index):
        return self.position[index]
    

    def index_of_position(self, position):
        x_index = round((position[0] - self.x_min) / self.pixel_size[0])
        y_index = round((position[1] - self.y_min) / self.pixel_size[1])

        if x_index > 0 and x_index < self.x_resolution and y_index > 0 and y_index < self.y_resolution:
            return x_index, y_index
        
        elif x_index < 0 and y_index < 0:
            return 0,0
        elif x_index >= self.x_resolution and y_index >= self.y_resolution:
            return self.x_resolution -1, self.y_resolution-1
        elif x_index >= self.x_resolution and y_index < 0:
            return self.x_resolution-1, 0
        elif x_index < 0 and y_index >= self.y_resolution:
            return 0, self.y_resolution-1

        
        elif x_index < 0:
            return 0, y_index
        elif x_index >= self.x_resolution:
            return self.y_resolution-1, y_index
        elif y_index < 0: 
            return x_index, 0
        elif y_index >= self.y_resolution:
            return x_index, self.y_resolution-1

        return 0,0

    def index_of_x_position(self, x_value):
        x_index = round((x_value - self.x_min) / self.pixel_size[0])

        if x_index < 0:
            return 0
        if x_index >= self.x_resolution:
            return self.x_resolution - 1
        return x_index
    
    def index_of_y_position(self, y_value):
        y_index = round((y_value - self.y_min) / self.pixel_size[1])

        if y_index < 0:
            return 0
        if y_index >= self.y_resolution:
            return self.y_resolution - 1
        return y_index
        
    
    def E_at_position(self, position):
        return self.E[self.index_of_position(position)]
    

    @time_it
    def calculate_e_field_numpy(self, charge):
        for old_index, old_position in enumerate(charge.old_positions):

            start = time.time()

            light_travel_distance = charge.light_travel_distance[old_index]

            center = self.index_of_position(old_position)
            outer_radius = light_travel_distance / self.pixel_size[0]
            inner_radius = (light_travel_distance - self.speed_of_light * self.dt) / self.pixel_size[0]

            mask = ((self.x - center[0])**2 + (self.y - center[1])**2 >= inner_radius**2) & ((self.x - center[0])**2 + (self.y - center[1])**2 <= outer_radius**2)

            if old_index == 0:
                self.E[(self.x - center[0])**2 + (self.y - center[1])**2 > outer_radius**2] = 0

            beta = (charge.old_velocities[old_index] / self.speed_of_light)[np.newaxis, :] #  1 x 3
            beta_point = (charge.old_accelerations[old_index] / self.speed_of_light)[np.newaxis, :]  # 1 x 3

            
            r_q = self.position[mask] - old_position[np.newaxis, :] # m x 3
            r_q_abs = np.sqrt(np.sum(r_q**2, axis=1))[:,np.newaxis] # m x 1
            e_q = r_q / r_q_abs # m x 3

            beta_squared = np.sum(beta*beta, axis=1) # 1 
            e_q_dot_beta = (np.sum(e_q * beta, axis=1))[:,np.newaxis] # m x 1
            e_q_dot_beta_point = (np.sum(e_q * beta_point, axis=1))[:,np.newaxis] # n x 1
            e_q_minus_beta = e_q - beta # m x 3

            E = charge.charge * (e_q_minus_beta * (1 - beta_squared) * r_q_abs**(-2) + (e_q_minus_beta*e_q_dot_beta_point - beta_point*(1-e_q_dot_beta)) * (r_q_abs*self.speed_of_light)**(-1)) * (1 - e_q_dot_beta)**(-3)
            # m x 3

            self.E[mask] = E
            self.E_norm[mask] = np.sqrt(np.sum(E**2, axis=1))

            print(f'took {(time.time() - start)*1000:.4g} ms')

        return

    @time_it
    def calculate_e_field_numexpr(self, charge):
        if np.size(charge.old_positions) == 0:
            return

        self.E *= 0
        
        for old_index, old_position in enumerate(charge.old_positions):
            light_travel_distance = charge.light_travel_distance[old_index]

            min_range_x = self.index_of_x_position(old_position[0] - 1.5*light_travel_distance)
            max_range_x = self.index_of_x_position(old_position[0] + 1.5*light_travel_distance)
            min_range_y = self.index_of_y_position(old_position[1] - 1.5*light_travel_distance)
            max_range_y = self.index_of_y_position(old_position[1] + 1.5*light_travel_distance)

            old_position = old_position[np.newaxis,np.newaxis,:] # 1 x 1 x 3

            r_q_part = ne.evaluate('pos - old_position', global_dict={'pos':self.position[min_range_x:max_range_x, min_range_y:max_range_y]}) # n x n x 3
            
            r_q_abs_part_1 = ne.evaluate('sum((r_q_part)**2, axis=2)') # n x n 

            r_q_abs_part = ne.evaluate('sqrt(r_q_abs_part_1)')

            beta = (charge.old_velocities[old_index] / self.speed_of_light)[np.newaxis, :] #  1 x 3
            beta_point = (charge.old_accelerations[old_index] / self.speed_of_light)[np.newaxis, :]  # 1 x 3

            relevant = np.where((r_q_abs_part <= light_travel_distance) & (r_q_abs_part > light_travel_distance - self.speed_of_light * self.dt)) # m elements

            if relevant[0].size == 0:
                continue
            
            r_q = r_q_part[relevant] # m x 3
            r_q_abs = (r_q_abs_part[relevant])[:,np.newaxis] # m x 1
            e_q = ne.evaluate('r_q / r_q_abs') # m x 3

            beta_squared = (ne.evaluate('sum(beta*beta, axis=1)')) # 1 
            e_q_dot_beta = (ne.evaluate('sum(e_q * beta, axis=1)'))[:,np.newaxis] # m x 1
            e_q_dot_beta_point = (ne.evaluate('sum(e_q * beta_point, axis=1)'))[:,np.newaxis] # n x 1
            e_q_minus_beta = ne.evaluate('e_q - beta') # m x 3

            E = ne.evaluate('(e_q_minus_beta * (1 - beta_squared) * r_q_abs**(-2) + (e_q_minus_beta*e_q_dot_beta_point - beta_point*(1-e_q_dot_beta)) * (r_q_abs*speed_of_light)**(-1)) * (1 - e_q_dot_beta)**(-3) ', global_dict={'speed_of_light':self.speed_of_light})
            # m x 3

            self.E[min_range_x:max_range_x, min_range_y:max_range_y][relevant] = ne.evaluate('charge * E', local_dict={'E':E,'charge':charge.charge})

        return


    def set_test_e_field(self, charge, charge_pos):
        vec_to_q = self.position - charge_pos
        self.E = charge * vec_to_q, 1 / (np.linalg.norm(vec_to_q, axis=2)**2)[:,:,np.newaxis]

    @time_it
    def E_field_in_color_numpy(self, saturation_point=1, scale_factor=1):
        # E_field_norm = np.sum(self.E**2, axis=2)

        E_field_color = np.tanh(self.E_norm / saturation_point)

        E_field_color = scipy.ndimage.gaussian_filter(E_field_color, 2)[:,:,np.newaxis]
        # E_field_color = (cv2.GaussianBlur(E_field_color, (9, 9),0))[:,:,np.newaxis]

        color_field = self.color_field*E_field_color

        color_field = np.repeat(np.repeat(color_field, scale_factor, axis=0), scale_factor, axis=1)

        return color_field


    @time_it
    def E_field_in_color_numexpr(self, saturation_point=1, scale_factor=1):
        E_field_norm = ne.evaluate('sum(E_field**2, axis=2)', global_dict={'E_field':self.E})

        E_field_color = ne.evaluate('tanh(E_field_norm**(0.5) / saturation_point)')

        # E_field_color = scipy.ndimage.gaussian_filter(E_field_color, 2)
        E_field_color = (cv2.GaussianBlur(E_field_color, (45, 45),0))[:,:,np.newaxis]

        color_field = ne.evaluate('color_field*E_field_color', global_dict={'color_field':self.color_field})

        color_field = np.repeat(np.repeat(color_field, scale_factor, axis=0), scale_factor, axis=1)

        return color_field


if __name__ == '__main__':


    a = np.array([[1,2,3],[4,5,6]])
    b = np.repeat(a, 5, axis=1)

    print(b)

    Space = Field_Area(600, 600, (-10, 10), (-10, 10), 0.05, 10)

    test_charge = Space.add_charge()

    for i in range(5):
        print(f'{i+1}. iteration')
        test_charge.set_update(Space.dt, np.array([0,np.sin(i),0]))
        Space.calculate_e_field_numpy(test_charge)
        Space.E_field_in_color_numpy()

