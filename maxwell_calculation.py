import numpy as np
import numexpr as ne
import dask.array as da
from multiprocessing import Pool


import time

def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} took {execution_time*1000:.3f} μs to execute.")
        return result
    return wrapper


BLUE = (0,0,255)
RED = (255,0,0)


def vectorfield_dot_product(vector_field, other_vector_field):
        return np.sum(vector_field * other_vector_field, axis=-1)
        
def vectorfield_scalarfield_product(vector_field, scalar_field):
    return vector_field * scalar_field[:,:,np.newaxis]

def vector_scalarfield_product(vector, scalar_field):
    return np.tile(vector, (*scalar_field.shape, 1)) * scalar_field[:,:,np.newaxis]

def add_vector_to_list(list, vector):
    return np.concatenate((list, vector[np.newaxis,:]))

class Charge():
    max_length_old = 20

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

        self.light_travel_distance = np.arange(self.max_length_old, 0, -1) * field.SPEED_OF_LIGHT * field.dt

    def lorentz_force(self):
        return self.charge *  (self.field.E_at_position(self.position) )#+ np.cross(self.velocity, np.array([0,0,0])))
    
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


class Fieldwave():
    SPEED_OF_LIGHT = 10
    def __init__(self, charge, origin, velocity, acceleration) -> None:
        self.charge = charge
        self.origin = origin
        self.charge_velocity = velocity / self.SPEED_OF_LIGHT
        self.charge_acceleration = acceleration / self.SPEED_OF_LIGHT
        self.radius = 0.1

    def update(self, dt):
        self.radius += dt*self.SPEED_OF_LIGHT

    def evaluate(self, position):
        e_q = (position - self.origin) / self.radius
        velocity_field = (e_q - self.charge_velocity) * (1 - np.linalg.norm(self.charge_velocity)**2) / (1 - np.dot(e_q, self.charge_velocity))**3 / self.radius**2
        acceleration_field = np.cross(e_q, np.cross(e_q-self.charge_velocity), self.charge_acceleration) / (self.SPEED_OF_LIGHT * self.radius * (1 - np.dot(e_q, self.charge_velocity))**3)
        return self.charge * (velocity_field + acceleration_field)
    

class Field_Area():

    SPEED_OF_LIGHT = 10
    dt = 0.05

    def __init__(self, x_resolution, y_resolution, x_range, y_range) -> None:
        self.E = np.zeros((x_resolution, y_resolution, 3))
        self.B = np.zeros((x_resolution, y_resolution, 3))

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

        self.charges = []

    def add_charge(self, **charge_properties):
        new_charge = Charge(self, **charge_properties)
        self.charges.append(new_charge)
        return new_charge


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
            return self.x_resolution, self.y_resolution
        elif x_index >= self.x_resolution and y_index < 0:
            return self.x_resolution, 0
        elif x_index < 0 and y_index >= self.y_resolution:
            return 0, self.y_resolution

        
        elif x_index < 0:
            return 0, y_index
        elif x_index >= self.x_resolution:
            return self.y_resolution, y_index
        elif y_index < 0: 
            return x_index, 0
        elif y_index >= self.y_resolution:
            return x_index, self.y_resolution

    def index_of_x_position(self, x_value):
        x_index = round((x_value - self.x_min) / self.pixel_size[0])

        if x_index < 0:
            return 0
        if x_index >= self.x_resolution:
            return self.x_resolution
        return x_index
    
    def index_of_y_position(self, y_value):
        y_index = round((y_value - self.y_min) / self.pixel_size[1])

        if y_index < 0:
            return 0
        if y_index >= self.y_resolution:
            return self.y_resolution
        return y_index
        
    
    def E_at_position(self, position):
        return self.E[self.index_of_position(position)]
    
    @time_it
    def calculate_e_field(self, charge):
        if np.size(charge.old_positions) == 0:
            return

        self.E *= 0
        
        for old_index, old_position in enumerate(charge.old_positions):
            light_travel_distance = charge.light_travel_distance[old_index]
            min_range_x = self.index_of_x_position(old_position[0] - light_travel_distance)
            max_range_x = self.index_of_x_position(old_position[0] + light_travel_distance)
            min_range_y = self.index_of_y_position(old_position[1] - light_travel_distance)
            max_range_y = self.index_of_y_position(old_position[1] + light_travel_distance)

            r_q_part = self.position[min_range_x:max_range_x, min_range_y:max_range_y] # n x n x 3
            r_q_abs_part = ne.evaluate('sum(r_q_part**2, axis=2)') # n x n 

            beta = (charge.old_velocities[old_index] / self.SPEED_OF_LIGHT)[np.newaxis, :] #  1 x 3
            beta_point = (charge.old_accelerations[old_index] / self.SPEED_OF_LIGHT)[np.newaxis, :]  # 1 x 3

            relevant = np.where((r_q_abs_part <= light_travel_distance) & (r_q_abs_part > light_travel_distance - self.SPEED_OF_LIGHT * self.dt)) # m elements

            if relevant[0].size == 0:
                continue

            r_q = r_q_part[relevant] # m x 3
            r_q_abs = (r_q_abs_part[relevant])[:,np.newaxis] # m x 1
            e_q = r_q / r_q_abs # m x 3

            beta_squared = (ne.evaluate('sum(beta*beta, axis=1)'))[:,np.newaxis] # 1 x 1 
            e_q_dot_beta = (ne.evaluate('sum(e_q * beta, axis=1)'))[:,np.newaxis] # m x 1
            e_q_dot_beta_point = (ne.evaluate('sum(e_q * beta_point, axis=1)'))[:,np.newaxis] # n x 1
            e_q_minus_beta = ne.evaluate('e_q - beta') # m x 3

            speed_of_light = self.SPEED_OF_LIGHT

            E = ne.evaluate('(e_q_minus_beta * (1 - beta_squared) * r_q_abs**(-2) + (e_q_minus_beta*e_q_dot_beta_point - beta_point*(1-e_q_dot_beta)) * (r_q_abs/speed_of_light)**(-1)) * (1 - e_q_dot_beta)**(-3)')
            # m x 3
            

            self.E[min_range_x:max_range_x, min_range_y:max_range_y][relevant] = E

        return

        r_q = np.tile(self.position, (len(charge.old_positions),1,1,1)) - charge.old_positions[:,np.newaxis,np.newaxis,:]

        r_q_abs = np.linalg.norm(r_q, axis=-1)

        e_q = r_q / r_q_abs[:,:,:,np.newaxis]

        beta = (charge.old_velocities / self.SPEED_OF_LIGHT)[:,np.newaxis,np.newaxis,:]
        beta_point = (charge.old_accelerations / self.SPEED_OF_LIGHT)[:,np.newaxis,np.newaxis,:]

        e_q_minus_beta = e_q - beta
        e_q_dot_beta = np.sum(e_q * beta, axis=-1)
        e_q_dot_beta_point = np.sum(e_q * beta_point, axis=-1)
        beta_squared = np.sum(beta*beta, axis=-1)

        # print(e_q)

        # print('heere')
        # print(beta_point * (1 - e_q_dot_beta))

        # print(e_q * beta)
        # print(e_q_minus_beta)


        relevant_points = np.where((r_q_abs <= light_travel_distance[:, np.newaxis, np.newaxis]) & (r_q_abs > light_travel_distance[:, np.newaxis, np.newaxis] -self.SPEED_OF_LIGHT * self.dt))

        # where_rel = np.where(relevant_points)
        # print(relevant_points)
        # print(where_rel)
        # print(where_rel[0])
        # print(r_q_abs[relevant_points])
        # print(where_rel[0])
        # print(r_q_abs[where_rel[0], where_rel[1]])
        # print(beta[relevant_points])

        beta_thing_pow_3 =  ((1 - e_q_dot_beta[relevant_points])**(-3))[:,np.newaxis]
        abc =  e_q_minus_beta[relevant_points]
        c = (1 - beta_squared[relevant_points[0]])[:,:,0]
        d = (r_q_abs[relevant_points]**2 )[:,np.newaxis]
        e = e_q_dot_beta_point[relevant_points][:,np.newaxis]
        f = beta_point[relevant_points[0]][:,0,0,:]
        h = (1-e_q_dot_beta[relevant_points])[:,np.newaxis]
        g = (r_q_abs[relevant_points])[:,np.newaxis]

        start_time = time.time()
        E = charge.charge *(beta_thing_pow_3 * abc * c / d + ((abc * e) - (f * h)) / g  / self.SPEED_OF_LIGHT)

        end_time = time.time()
        print(f'calc r_q took {(end_time - start_time)*1000:.4f} μs')
        
        # print(self.E[relevant_points[1:]])

        self.E = np.zeros_like(self.E)
        self.E[relevant_points[1:]] = E
        # self.E[relevant_points] = charge.charge * (1 - e_q_dot_beta[relevant_points])**(-3) * ( (e_q_minus_beta[relevant_points]) * (1 - beta**2) / r_q_abs[relevant_points]**2 + (e_q_minus_beta[relevant_points] * e_q_dot_beta_point[relevant_points] - beta_point * (1-e_q_dot_beta[relevant_points])) / r_q_abs[relevant_points] / self.SPEED_OF_LIGHT )

        # print(light_travel_distance)
        # print(r_q_abs < light_travel_distance[:, np.newaxis, np.newaxis] * 100)
        # print(r_q_abs[np.where(r_q_abs < light_travel_distance[:, np.newaxis, np.newaxis] * 100)])
    

    def set_test_e_field(self, charge, charge_pos):
        vec_to_q = self.position - charge_pos
        self.E = charge * vectorfield_scalarfield_product(vec_to_q, 1 / np.linalg.norm(vec_to_q, axis=2)**2)


    def E_field_in_color(self, color_positive=RED, color_negative=BLUE, saturation_point=1, x_range='full', y_range='full'):
        color_field = np.zeros(self.E.shape, dtype=np.uint8)
        E_field = self.E / saturation_point
        E_field_norm = np.linalg.norm(E_field, axis=2)

        E_field_to_big = np.where(E_field_norm >= 1)
        rest = np.where(E_field_norm < 1)

        color_field[E_field_to_big] = BLUE
        color_field[rest] = (np.tile(BLUE, (E_field_norm[rest].shape[0], 1)) * E_field_norm[rest][:,np.newaxis]).round()
        

        color_field = np.tile(BLUE, (*E_field_norm.shape, 1)) * np.tanh(E_field_norm)[:,:,np.newaxis]

        return color_field

        # print(np.where(E_field >= 1))

        # a = np.array([[1,0,0,1],[0,0,1,2]])
        # print((a>1) &( a<=0))
        # print([*a.shape, 15])

        # color_field[E_field_norm >= 1] = color_positive
        # color_field[E_field_norm <= -1] = color_negative

        # small_positive = (E_field_norm < 1) & (E_field_norm >=0)
        # small_E_positive = E_field_norm[np.where(small_positive)]
        # color_field[small_positive] = (small_E_positive.reshape(len(small_E_positive), 1) @ np.array(color_positive).reshape(1,3)).round()
        
        # small_negative = (E_field_norm > -1) & (E_field_norm < 0)
        # small_E_negative = np.abs(E_field_norm[np.where(small_negative)])
        # color_field[small_negative] = (small_E_negative.reshape(len(small_E_negative), 1) @ np.array(color_negative).reshape(1,3)).round()

        # return color_field
    


# a = np.linspace(0,10,6)
# b = np.linspace(0,10,5)

# c = np.tile(a, (5,1))
# d = np.tile(b, (6,1)).T

# e = np.stack((c,d,np.zeros_like(c)), axis=-1)

# f = np.array([[1,2,3], [1,1,1]])
# h = np.array([1,2,3])

# print(np.tile(f, (2, 1,1)))

# print(c)
# print(c[1:4,2])
# # print(d)
# # print()
# print(e)
# print('rntrn')
# print(e[1:5,1:4].reshape(-1, 3))
# print('here')
# print(vector_scalarfield_product(f, c))
    

# print('here')
# print(vector_dot_product(e, e))

# print(b.shape)

if __name__ == '__main__':

    Space = Field_Area(600, 600, (-10, 10), (-10, 10))

    test_charge = Space.add_charge()


    # a = np.random.rand(20*600*600*3)
    # b = np.random.rand(20*600*600*3)

    # c = np.random.rand(20*600*600*3)
    # d = np.random.rand(20*600*600*3)

    # def multiply_chunk(chunk):
    #     return chunk[0] * chunk[1]

    # array1_chunks = np.array_split(a, 4)
    # array2_chunks = np.array_split(b, 4)

    # print(np.tile(np.array([1,2,3]), (5,1)))


    # start_time = time.time()
    # for _ in range(10):
        # with Pool(processes=4) as pool:
        #     result_chunks = pool.map(multiply_chunk, zip(array1_chunks, array2_chunks))

        # result = np.concatenate(result_chunks)
# 
        # c = ne.evaluate('a*b/a + c**2/d + a*b/a + c**2/d')
        # r = a*b/a + c**2/d + a*b/a + c**2/d
        # r = np.sum(a*b, axis=3)
        # c = ne.evaluate('sum(a*a, axis=3) + sum(a*a, axis=3)')
        # r = np.multiply(c,d, out=c)
        # c = da.multiply(a,b)
    # end_time = time.time()
    # print(f'calc r_q took {(end_time - start_time)*1000:.4f} μs')
        

    for i in range(6):
        print(f'{i+1}. iteration')
        test_charge.set_update(Space.dt, np.array([0,np.sin(i),0]))
        Space.calculate_e_field(test_charge)
        Space.E_field_in_color()

