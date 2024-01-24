import numpy as np

BLUE = (0,0,255)
RED = (255,0,0)

class Charge():
    def __init__(self, charge=1, mass=1, init_position=np.zeros(3), init_velocity=np.zeros(3)) -> None:
        self.charge = charge
        self.mass = mass
        self.position = init_position
        self.velocity = init_velocity
        self.acceleration = np.zeros(3)

    def x(self):
        return self.position[0]
    def y(self):
        return self.position[1]
    def z(self):
        return self.position[2]

    def lorentz_force(self, E, B):
        return self.charge * (E[int(self.x()), int(self.y()),:] + np.cross(self.velocity, np.array([0,0,0])))
    

    def update(self, dt, E, B):
        self.acceleration = self.lorentz_force(E, B) / self.mass
        self.position += self.velocity * dt
        self.velocity += self.acceleration * dt

    def set_update(self, dt, pos):
        old_pos = self.position
        old_vel = self.velocity
        self.position = pos
        self.velocity = (self.position - old_pos) / dt
        self.acceleration = (self.velocity - old_vel) / dt


    


class Fieldwave():
    SPEED_OF_LIGHT = 1
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
    

class BliBlaBlubb():
    def __init__(self, width, height, x_range, y_range) -> None:
        self.E = np.ones((width, height, 3))
        i, j = np.indices((width, height), dtype=int)
        self.E_indices = np.concatenate([i[:, :, np.newaxis], j[:, :, np.newaxis], np.zeros_like(i)[:, :, np.newaxis]], axis=-1)

        self.B = np.zeros((width, height, 3))
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.x_width = self.x_max - self.x_min
        self.y_width = self.y_max - self.y_min
        self.pixel_size = (self.x_width / (width - 1), self.y_width / (height - 1))
        self.origin = (width/2, height/2)
        self.charges = None


    def static_electric_field(pos):
        return np.linalg.norm(pos) ** (-2)
    
    def index_to_coordinates(self, index):
        return np.array([self.x_min + index[0]*self.pixel_size[0], self.y_max - index[1]*self.pixel_size[1], np.zeros_like(index[0])])
    
    def set_test_e_field(self, charge, charge_idx):
        positions = self.index_to_coordinates(self.E_indices)
        charge_pos = self.index_to_coordinates(charge_idx)

        print(positions[0,0])


        # a = np.array([[[1, 1, 1], [0, 0, 0]],
        #       [[2, 2, 2], [3, 3, 3]]])

        # b = np.array([1, 2, 3])

        # print(b.reshape(1,1,3))

        # # Use broadcasting to add each element of b to the corresponding element in a
        # result = a + b.reshape(1,1,3)

        # print(result)
        # print('over here')

        # a = np.array([[[1,1,1], [0,0,0]], 
        #               [[2,2,2], [3,3,3]]])
        
        # b = np.array([1,2,3])

        # # for i in a:
        # #     for j in i:
        # #         j += b

        # print(a)

        # print('here')
        # # print(a - b[:, np.newaxis])
        # print(b[:, np.newaxis])
        # print(a - b[np.newaxis,np.newaxis, :])

        vec_to_q = positions - charge_pos[np.newaxis, np.newaxis,:]
        self.E = charge * (vec_to_q) / np.linalg.norm(vec_to_q, axis=2)


    def E_field_in_color(self, color_positive=RED, color_negative=BLUE, saturation_point=1, x_range='full', y_range='full'):
        color_field = np.zeros(self.E.shape, dtype=np.uint8)
        E_field = self.E / saturation_point
        E_field_norm = np.linalg.norm(E_field, axis=2)

        # print(np.where(E_field >= 1))

        # a = np.array([[1,0,0,1],[0,0,1,2]])
        # print((a>1) &( a<=0))
        # print([*a.shape, 15])

        color_field[E_field_norm >= 1] = color_positive
        color_field[E_field_norm <= -1] = color_negative

        small_positive = (E_field_norm < 1) & (E_field_norm >=0)
        small_E_positive = E_field_norm[np.where(small_positive)]
        color_field[small_positive] = (small_E_positive.reshape(len(small_E_positive), 1) @ np.array(color_positive).reshape(1,3)).round()
        
        small_negative = (E_field_norm > -1) & (E_field_norm < 0)
        small_E_negative = np.abs(E_field_norm[np.where(small_negative)])
        color_field[small_negative] = (small_E_negative.reshape(len(small_E_negative), 1) @ np.array(color_negative).reshape(1,3)).round()

        return color_field
    



baum = BliBlaBlubb(5, 5, (-10, 10), (-10, 10))

baum.set_test_e_field(1, (2,3))
baum.E_field_in_color()

