import time

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
    max_length_old = 80

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

        self.light_travel_distance = np.arange(self.max_length_old, -1, -1) * field.speed_of_light * field.dt
        self.old_dt = np.ones(self.max_length_old) * field.dt


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


    def set_update(self, dt, pos, vel=None, acc=None):
        self.old_positions = add_vector_to_list(self.old_positions, self.position)
        self.old_velocities = add_vector_to_list(self.old_velocities, self.velocity)
        self.old_accelerations = add_vector_to_list(self.old_accelerations, self.acceleration)
        while self.old_positions.shape[0] > self.max_length_old:
            self.old_positions = np.delete(self.old_positions, 0, axis=0)
            self.old_velocities = np.delete(self.old_velocities, 0, axis=0)
            self.old_accelerations = np.delete(self.old_accelerations, 0, axis=0)

        self.light_travel_distance += self.field.dx
        self.light_travel_distance = np.concatenate((np.delete(self.light_travel_distance, 0), [0]))

        old_pos = self.position
        old_vel = self.velocity
        self.position = pos
        if vel is None:
            self.velocity = (self.position - old_pos) / dt
        else:
            self.velocity = vel
        if acc is None:
            self.acceleration = (self.velocity - old_vel) / dt
        else:
            self.acceleration = acc



class Field_Area():
    def __init__(self, x_resolution, y_resolution, pixel_size, dt, c) -> None:
        self.E = np.zeros((x_resolution, y_resolution, 3))
        self.E_norm = np.zeros((x_resolution, y_resolution))
        self.B = np.zeros((x_resolution, y_resolution, 3))

        self.x, self.y = np.indices((x_resolution, y_resolution))
        self.center = np.array([x_resolution//2, y_resolution//2])

        self.speed_of_light = c
        self.dt = dt
        self.dx = self.dt * self.speed_of_light

        self.pixel_size = pixel_size
        self.diagonal = (x_resolution**2 + y_resolution**2)**(0.5)

        self.x_resolution = x_resolution
        self.y_resolution = y_resolution

        self.x_width = self.pixel_size*(x_resolution - 1)
        self.y_width = self.pixel_size*(y_resolution - 1)

        self.x_max = self.pixel_size*(x_resolution - 1) / 2
        self.x_min = -self.x_max
        self.y_max = self.pixel_size*(y_resolution - 1) / 2
        self.y_min = -self.y_max


        x = np.linspace(self.x_min, self.x_max, x_resolution)
        y = np.linspace(self.y_min, self.y_max, y_resolution)
        x_grid = np.tile(x, (y_resolution, 1)).T
        y_grid = np.tile(y, (x_resolution, 1))
        z_grid = np.zeros_like(x_grid)

        self.position = np.stack((x_grid, y_grid, z_grid), axis=-1)
        # self.indices = np.stack(np.indices((x_resolution, y_resolution)), axis=-1)

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
    
    def set_dt(self, new_dt):
        self.dt = new_dt
        self.dx = self.speed_of_light*self.dt


    def position_at_index(self, index, scale_factor):
        return np.array([index[0] * self.pixel_size / scale_factor + self.x_min, index[1] * self.pixel_size / scale_factor + self.y_min, 0])
    

    def index_of_position(self, position, scale_factor=1):
        x_index = round((position[0] - self.x_min) / self.pixel_size * scale_factor)
        y_index = round((position[1] - self.y_min) / self.pixel_size * scale_factor)

        scaled_x_res = self.x_resolution * scale_factor
        scaled_y_res = self.y_resolution * scale_factor

        if x_index > 0 and x_index < scaled_x_res and y_index > 0 and y_index < scaled_y_res:
            return np.array([x_index, y_index])
        
        elif x_index < 0 and y_index < 0:
            return np.array([0, 0])
        elif x_index >= scaled_x_res and y_index >= scaled_y_res:
            return np.array([scaled_x_res -1, scaled_y_res-1])
        elif x_index >= scaled_x_res and y_index < 0:
            return np.array([scaled_x_res-1, 0])
        elif x_index < 0 and y_index >= scaled_y_res:
            return np.array([0, scaled_y_res-1])

        
        elif x_index < 0:
            return np.array([0, y_index])
        elif x_index >= scaled_x_res:
            return np.array([scaled_x_res-1, y_index])
        elif y_index < 0: 
            return np.array([x_index, 0])
        elif y_index >= scaled_y_res:
            return np.array([x_index, scaled_y_res-1])

        return np.array([0,0])

    def index_of_x_position(self, x_value):
        x_index = round((x_value - self.x_min) / self.pixel_size)

        if x_index < 0:
            return 0
        if x_index >= self.x_resolution:
            return self.x_resolution - 1
        return x_index
    
    def index_of_y_position(self, y_value):
        y_index = round((y_value - self.y_min) / self.pixel_size)

        if y_index < 0:
            return 0
        if y_index >= self.y_resolution:
            return self.y_resolution - 1
        return y_index
        
    
    def E_at_position(self, position):
        return self.E[tuple(self.index_of_position(position))]
    

    # @time_it
    def calculate_e_field_numpy(self, charge):
        for old_index, old_position in enumerate(charge.old_positions):
            center = self.index_of_position(old_position)
            outer_radius = (charge.light_travel_distance[old_index] / self.pixel_size).astype(np.float32)
            inner_radius = (charge.light_travel_distance[old_index+1] / self.pixel_size).astype(np.float32)

            if (self.center[0] - center[0])**2 + (self.center[1] - center[1])**2 + self.diagonal/2 < inner_radius:
                continue

            index_distance = (self.x - center[0])**2 + (self.y - center[1])**2
            mask = (index_distance>= inner_radius**2) & (index_distance <= outer_radius**2)

            if old_index == 0:
                outside = (self.x - center[0])**2 + (self.y - center[1])**2 > outer_radius**2
                self.E[outside] = np.zeros(3)
                self.E_norm[outside] = 0

            beta = (charge.old_velocities[old_index] / self.speed_of_light)[np.newaxis, :] #  1 x 3
            beta_point = (charge.old_accelerations[old_index] / self.speed_of_light)[np.newaxis, :]  # 1 x 3

            r_q = self.position[mask] - old_position[np.newaxis, :] # m x 3
            r_q_abs = np.sqrt(np.sum(r_q**2, axis=1), dtype=np.float32)[:,np.newaxis] # m x 1
            e_q = r_q / r_q_abs # m x 3

            beta_squared = np.sum(beta*beta, axis=1, dtype=np.float32) # 1 
            e_q_dot_beta = (np.sum(e_q * beta, axis=1, dtype=np.float32))[:,np.newaxis] # m x 1
            e_q_dot_beta_point = (np.sum(e_q * beta_point, axis=1, dtype=np.float32))[:,np.newaxis] # n x 1
            e_q_minus_beta = e_q - beta # m x 3

            E = charge.charge * (e_q_minus_beta * (1 - beta_squared) * r_q_abs**(-2) + (e_q_minus_beta*e_q_dot_beta_point - beta_point*(1-e_q_dot_beta)) * (r_q_abs*self.speed_of_light)**(-1)) * (1 - e_q_dot_beta)**(-3)
            # m x 3

            self.E[mask] = E
            self.E_norm[mask] = np.sqrt(np.sum(E**2, axis=1))
        return


    def set_test_e_field(self, charge, charge_pos):
        vec_to_q = self.position - charge_pos
        self.E = charge * vec_to_q, 1 / (np.linalg.norm(vec_to_q, axis=2)**2)[:,:,np.newaxis]


    # @time_it
    def E_field_in_color_numpy(self, saturation_point=1, scale_factor=1):
        E_field_color = np.tanh(np.sqrt(self.E_norm / saturation_point), dtype=np.float32)

        E_field_color = scipy.ndimage.gaussian_filter(E_field_color, 2)[:,:,np.newaxis]

        color_field = (self.color_field*E_field_color).astype(np.uint8)

        color_field = np.repeat(np.repeat(color_field, scale_factor, axis=0), scale_factor, axis=1)

        return color_field

if __name__ == '__main__':

    a = np.array([[1,2,3],[4,5,6]], dtype=np.float16)
    b = a / 100000
    print(a)
    print(b)

    Space = Field_Area(900//2, 600//2, 0.15, 0.01, 10)

    test_charge = Space.add_charge()

    for i in range(2):
        print(f'{i+1}. iteration')
        test_charge.set_update(Space.dt, np.array([0,np.sin(i),0]))
        Space.calculate_e_field_numpy(test_charge)
        Space.E_field_in_color_numpy(scale_factor=3)

