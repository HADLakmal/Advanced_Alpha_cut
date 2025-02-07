import numpy as np

def findCentroids(shortestDistances, K):
    N = shortestDistances.shape[0]
    centroids = np.random.choice(N, size=K, replace=False)

    iterations = 0
    MAX_ITERS = 15
    converged = False
    while not converged and iterations < MAX_ITERS:
        iterations+=1
        print("STEP",iterations)
        print("  Centroids:",centroids)

        # Associate every node the centroid that has the minimum
        # travel distance to that node
        shortestDistancesToCentroids = np.full((N,N), np.iinfo(int).max)
        shortestDistancesToCentroids[centroids,:] = shortestDistances[centroids,:] * demands.transpose()
        nearestCentroid = np.argmin(shortestDistancesToCentroids, axis=0)

        # TODO: find a nice numpy one liner to create the clusters
        clusters = [ [n for n in range(N) if nearestCentroid[n] == c] for c in centroids]
        newCentroids = []

        print("  Clusters:")
        for k in clusters:
            notInK = [n for n in range(N) if n not in k]
            sd_per_dem_clusters = sd_per_dem.copy()
            sd_per_dem_clusters[:,notInK] = 0
            totalCostToOtherNodesInCluster = np.sum(sd_per_dem_clusters, axis = 1)
            totalCostToOtherNodesInCluster[notInK] = np.iinfo(int).max
            centroid = np.argmin(totalCostToOtherNodesInCluster)
            print("    ", k, "->", centroid)
            newCentroids.append(centroid)

        converged = np.array_equal(centroids, newCentroids)
        centroids = newCentroids

    return centroids

# To avoid falling in a suboptimal minimum run the centroid search many times and
# count how many times a node is selected as a centroid
def findCentroidsVoting(shortestDistances, K):
    N = shortestDistances.shape[0]
    votes = np.zeros(N, dtype=int)
    for t in range(K*N):
        centroids = findCentroids(shortestDistances, K)
        votes[centroids]+=1
    print("Votes:", votes)
    return np.argpartition(votes, -K)[-K:]


if __name__ == '__main__':
    N = 7
    K = 2

    adjacencyMatrix = np.array([
        [ 0, 2, 2,-1,-1,-1,-1],
        [ 2, 0, 3,-1,-1,-1,-1],
        [ 2, 3, 0, 4,-1,-1,-1],
        [-1,-1, 4, 0, 3,-1,-1],
        [-1,-1,-1, 3, 0, 5, 2],
        [-1,-1,-1,-1, 5, 0, 8],
        [-1,-1,-1,-1, 2, 8, 0]
    ])
    demands = np.array([1, 3, 6, 7, 1, 4, 3])

    shortestDistances = np.array([
        [-0.98518419,-0.17149958]
        [-0.98518419,-0.17149958]
        [-0.25049594,0.96811765]
        [0.99749775,0.07069817]
        [-0.1015989 ,-0.99482544]
        [0.99878303,0.04932004]
        [0.99749775,0.07069817]
        [0.99880343,0.04890516]
        [0.99880343,0.04890516]
        [0.99880343,0.04890516]])


    sd_per_dem = shortestDistances * demands

    print("Shortest distances:\n",shortestDistances)
    print("Node demands:",demands)
    print("Weighted distances:\n",sd_per_dem)

    centroids = findCentroidsVoting(shortestDistances, K)

    #[1 1 1 0 1 0 0 0 0 0]