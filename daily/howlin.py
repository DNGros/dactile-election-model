import numpy as np
import statsmodels.stats.correlation_tools

from state_correlations import load_five_thirty_eight_correlation, correlation_dict_to_matrix, \
    load_economist_correlations



def is_positive_semidefinite(M):
    #for _ in range(1000):
    #    z = np.random.randn(M.shape[0])
    #    result = z.T @ M @ z
    #    if result < 0:
    #        return False
    eigenvalues = np.linalg.eigvals(M)
    return np.all(eigenvalues >= 0)


def main():

    # should be psd
    M1 = np.array([[2, -1, 0],
                   [-1, 2, -1],
                   [0, -1, 2]])

    print("M1 is positive semidefinite:", is_positive_semidefinite(M1))

    M2 = np.array([[1, 2],
                   [2, 1]])

    print("M2 is positive semidefinite:", is_positive_semidefinite(M2))

    # Demonstrating z^T M z for a specific vector
    z = np.array([2, 1, 1])
    print(f"{z.T=}")
    print(f"{z.T @ M1=}")
    result_M1 = z.T @ M1 @ z
    print("z^T M1 z =", result_M1)

    z = np.array([1, -1])
    result_M2 = z.T @ M2 @ z
    print("z^T M2 z =", result_M2)

    correlation_matrix = np.array([
        [1, 0.2, 0.8],
        [0.2, 1, 0.0],
        [0.8, 0.0, 1]
    ])
    print("correlation_matrix is positive semidefinite:", is_positive_semidefinite(correlation_matrix))
    eigenvalues = np.linalg.eigvals(correlation_matrix)
    print("Eigenvalues of matrix:", eigenvalues)
    sample_val = np.array([3, 0, 0])
    print(f"{sample_val.T=}")
    print(f"{sample_val.T @ correlation_matrix=}")
    print(f"{sample_val.T @ correlation_matrix @ sample_val=}")

    assert np.all(eigenvalues >= 0)
    assert is_positive_semidefinite(correlation_matrix)

    def adjusted_corr(correlation_matrix, row, value):
        # start with identity matrix
        v = np.eye(correlation_matrix.shape[0])
        # adjust the row
        v[row] = correlation_matrix[row]**value
        # adjust the column
        v[:, row] = correlation_matrix[:, row]**value
        print("ident\n", v)
        print("Is psd?", is_positive_semidefinite(v))
        # make psd
        #return get_near_psd(v)
        return v

    new_corr = adjusted_corr(correlation_matrix, 0, 2)
    print("New correlation matrix:\n", new_corr)
    print("Is new correlation matrix positive semidefinite:", is_positive_semidefinite(new_corr))

    #nearest = get_near_psd(correlation_matrix)
    #print("Nearest positive semidefinite matrix:\n", nearest)
    #print("Is nearest positive semidefinite matrix:", is_positive_semidefinite(nearest))

    mat, states = correlation_dict_to_matrix(load_five_thirty_eight_correlation())
    #mat, states = correlation_dict_to_matrix(load_economist_correlations())
    print(mat)
    print("is 538 psd", is_positive_semidefinite(mat))
    pa_index = states.index("PA")
    new_corr = adjusted_corr(mat, pa_index, 3)
    with np.printoptions(threshold=100*100, linewidth=100000, suppress=True):
        print("New 538 correlation matrix:\n", new_corr)
        if not is_positive_semidefinite(new_corr):
            print("NOT psd")
            print("Nearest psd matrix")
            print(statsmodels.stats.correlation_tools.cov_nearest(new_corr))


if __name__ == "__main__":
    main()
