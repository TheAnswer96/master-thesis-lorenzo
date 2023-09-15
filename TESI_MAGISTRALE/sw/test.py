from graph import *

n_scenarios = 100

if __name__ == '__main__':
    n = 25
    for c in [0.5, 1, 1.5, 2]:
        print("c=%.1f" % c)
        diameters = np.zeros((n_scenarios))
        degrees = np.zeros((n_scenarios))
        for i in range(0, n_scenarios):
            G, pos = create_graph(c)

            diameters[i] = nx.diameter(G)

            sum = 0
            for j in range(0, n):
                sum = sum + G.degree[j]

            degrees[i] = sum/n

        avg_diam = np.mean(diameters, axis=0)
        std_diam = np.std(diameters, axis=0)
        print("diam => avg=%.2f, std=%.2f" % (avg_diam, std_diam))

        avg_deg = np.mean(degrees, axis=0)
        std_deg  = np.std(degrees, axis=0)
        print("degr => avg=%.2f, std=%.2f" % (avg_deg , std_deg ))
