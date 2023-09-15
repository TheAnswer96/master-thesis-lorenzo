from model import *
import csv


def default_parameters():
    distance = 1
    payload_weight = 0.0
    drone_speed = 10
    wind_speed = 0
    wind_direction = 0

    return distance, payload_weight, drone_speed, wind_speed, wind_direction


def write_output(data, label):
    out_file = "out/test/test-%s.csv" % label
    with open(out_file, 'w', newline='') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(data)
    writeFile.close()


def test_distances():
    distance, payload_weight, drone_speed, wind_speed, wind_direction = default_parameters()

    n = 41
    data = np.zeros((n, 2))
    i = 0
    for distance in np.linspace(0, 5000, num=n):
        E = get_energy(distance, payload_weight, drone_speed, wind_speed, wind_direction)
        data[i, 0] = round(distance, 2)
        data[i, 1] = round(E, 2)
        i = i+1

    write_output(data, "distances")


def test_payloads():
    distance, payload_weight, drone_speed, wind_speed, wind_direction = default_parameters()

    n = 41
    data = np.zeros((n, 2))
    i = 0
    for payload_weight in np.linspace(0, 7, num=n):
        E = get_energy(distance, payload_weight, drone_speed, wind_speed, wind_direction)
        data[i, 0] = round(payload_weight, 2)
        data[i, 1] = round(E, 2)
        i = i+1

    write_output(data, "payloads")


def test_speeds():
    distance, payload_weight, drone_speed, wind_speed, wind_direction = default_parameters()

    n = 41
    data = np.zeros((n, 7))
    i = 0
    for drone_speed in np.linspace(1, 20, num=n):
        #    get_energy(distance, payload_weight, drone_speed, wind_speed, wind_direction)
        E1 = get_energy(distance, 0, drone_speed, 0, 0)
        E2 = get_energy(distance, 0, drone_speed, 10, 0)
        E3 = get_energy(distance, 0, drone_speed, 10, 180)

        E4 = get_energy(distance, 7, drone_speed, 0, 0)
        E5 = get_energy(distance, 7, drone_speed, 10, 0)
        E6 = get_energy(distance, 7, drone_speed, 10, 180)

        data[i, 0] = round(drone_speed, 2)
        data[i, 1] = round(E1, 2)
        data[i, 2] = round(E2, 2)
        data[i, 3] = round(E3, 2)
        data[i, 4] = round(E4, 2)
        data[i, 5] = round(E5, 2)
        data[i, 6] = round(E6, 2)

        i = i+1

    write_output(data, "speeds")


def test_wind_head():
    distance, payload_weight, drone_speed, wind_speed, wind_direction = default_parameters()

    n = 41
    data = np.zeros((n, 2))
    i = 0
    wind_direction = 180
    for wind_speed in np.linspace(0, 10, num=n):
        E = get_energy(distance, payload_weight, drone_speed, wind_speed, wind_direction)
        data[i, 0] = round(wind_speed, 2)
        data[i, 1] = round(E, 2)
        i = i+1

    write_output(data, "wind_head")


def test_wind_tail():
    distance, payload_weight, drone_speed, wind_speed, wind_direction = default_parameters()

    n = 41
    data = np.zeros((n, 2))
    i = 0
    wind_direction = 0
    for wind_speed in np.linspace(0, 10, num=n):
        E = get_energy(distance, payload_weight, drone_speed, wind_speed, wind_direction)
        data[i, 0] = round(wind_speed, 2)
        data[i, 1] = round(E, 2)
        i = i+1

    write_output(data, "wind_tail")


if __name__ == '__main__':

    # tests used only for the plots
    # test_distances()
    # test_payloads()
    test_speeds()
    # test_wind_head()
    # test_wind_tail()
