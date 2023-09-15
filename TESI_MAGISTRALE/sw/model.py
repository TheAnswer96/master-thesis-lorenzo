from sympy import Symbol, nsolve
import sympy as sp
import numpy as np
import mpmath

mpmath.mp.dps = 5


def get_energy(distance, payload_weight, drone_speed, wind_speed, wind_direction):

    # start calculations
    m_package = payload_weight
    m_drone = 7
    m_battery = 10

    num_rotors = 8
    diameter = 0.432

    # s_battery = 540000
    # delta = 0.5
    # f = 1.2

    pressure = 100726  # 50 meters above sea level
    R = 287.058
    temperature = 15 + 273.15  # 15 degrees in Kelvin
    rho = pressure / (R*temperature)

    g = 9.81

    # power efficiency
    eta = 0.7

    drag_coefficient_drone = 1.49
    drag_coefficient_battery = 1
    drag_coefficient_package = 2.2

    projected_area_drone = 0.224
    projected_area_battery = 0.015
    projected_area_package = 0.0929

    v_north = drone_speed - wind_speed*np.cos(np.deg2rad(wind_direction))
    v_east = - wind_speed*np.sin(np.deg2rad(wind_direction))
    v_air = np.sqrt(v_north**2 + v_east**2)

    # Drag force
    F_drag_drone = 0.5 * rho * (v_air**2) * drag_coefficient_drone * projected_area_drone
    F_drag_battery = 0.5 * rho * (v_air**2) * drag_coefficient_battery * projected_area_battery
    F_drag_package = 0.5 * rho * (v_air**2) * drag_coefficient_package * projected_area_package

    F_drag = F_drag_drone + F_drag_battery + F_drag_package

    alpha = np.arctan(F_drag / ((m_drone + m_battery + m_package)*g))

    # Thrust
    T = (m_drone + m_battery + m_package)*g + F_drag

    # # Power min hover
    # P_min_hover = (T**1.5) / (np.sqrt(0.5 * np.pi * num_rotors * (diameter**2) * rho))

    # v_i = Symbol('v_i')
    # f_0 = v_i - (2*T / (np.pi * num_rotors * (diameter**2) * rho * sp.sqrt((drone_speed*sp.cos(alpha))**2 + (drone_speed*sp.sin(alpha) + v_i)**2)))
    # induced_speed = float(nsolve(f_0, v_i, 5))
    # print(induced_speed)

    tmp_a = 2*T
    tmp_b = np.pi * num_rotors * (diameter**2) * rho
    tmp_c = (drone_speed*sp.cos(alpha))**2
    tmp_d = drone_speed*sp.sin(alpha)
    tmp_e = tmp_a / tmp_b

    coeff = [1, (2*tmp_d), (tmp_c+tmp_d**2), 0, -tmp_e**2]
    sol = np.roots(coeff)
    induced_speed = float(max(sol[np.isreal(sol)]).real)
    # print(induced_speed)

    # Power min to go forward
    P_min = T*(drone_speed*np.sin(alpha) + induced_speed)

    # expended power
    P = P_min / eta

    # energy efficiency of travel
    mu = P / drone_speed

    # Energy consumed
    E = mu * distance

    # # Range of a drone
    # R = (m_battery * s_battery * delta) / (e * f)

    return E/1000.


def compute_prefixes():
    distance = 1
    payload_weights = [0, 2]
    drone_speeds = [10, 20]
    global_wind_speeds = [0, 5, 10, 15]
    relative_wind_directions = [0, 45, 135, 180]

    prefix = {}
    for p in payload_weights:
        for v in drone_speeds:
            for ws in global_wind_speeds:
                for wd in relative_wind_directions:
                    prefix[(p, v, ws, wd)] = get_energy(distance, p, v, ws, wd)

    return prefix


if __name__ == '__main__':
    prefix = compute_prefixes()
    print(prefix)


