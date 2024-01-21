import numpy as np

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
        return self.charge * (np.array([0,0,0]) + np.cross(self.velocity, np.array([0,0,1])))
    

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
    def __init__(self, origin, velocity, acceleration) -> None:
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
    def __init__(self) -> None:
        self.E = None
        self.B = None
        self.charges = None


    

