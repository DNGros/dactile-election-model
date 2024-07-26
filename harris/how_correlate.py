from state_correlations import corr2cov, load_five_thirty_eight_correlation, get_multivariate_t_dist, \
    calc_scale_factor_for_t_dist
import numpy as np


def old_magic():
    import numpy as np
    from scipy import stats

    def simulate_correlated_polling_misses(correlation_dict, mean_miss=2, std_dev=2, num_simulations=1):
        # Get the list of states
        states = list(correlation_dict.keys())

        # Convert the correlation dictionary to a matrix
        correlation_matrix = np.array(
            [[correlation_dict[state1][state2] for state2 in states] for state1 in states])

        cov_matrix = corr2cov(correlation_matrix, std_dev)

        # Generate samples from a multivariate normal distribution
        samples = stats.multivariate_normal.rvs(mean=np.zeros(len(states)),
                                                cov=cov_matrix,
                                                size=num_simulations)

        # Scale the samples to match our desired mean and standard deviation
        #scaled_samples = (samples * std_dev) + mean_miss
        scaled_samples = samples + mean_miss

        # Convert the results to a dictionary
        result = {state: scaled_samples[:, i] for i, state in enumerate(states)}

        return result

    # Example usage with verification:
    #correlation_dict = {
    #    'AK': {'AK': 1.0, 'AL': 0.3125730782481495, 'AR': 0.44585450409115535},
    #    'AL': {'AK': 0.3125730782481495, 'AL': 1.0, 'AR': 0.6248974592250378},
    #    'AR': {'AK': 0.44585450409115535, 'AL': 0.6248974592250378, 'AR': 1.0}
    #}
    correlation_dict = load_five_thirty_eight_correlation()

    mean_miss = 0.00
    std_dev = 0.02
    num_simulations = 10000


    simulated_misses = simulate_correlated_polling_misses(correlation_dict, mean_miss, std_dev,
                                                          num_simulations)

    # Verify the mean and standard deviation for each state
    print("Verification of mean and standard deviation:")
    for state, misses in simulated_misses.items():
        print(f"{state}:")
        print(f"  Mean: {np.mean(misses):.4f} (Expected: {mean_miss})")
        print(f"  Std Dev: {np.std(misses):.4f} (Expected: {std_dev})")
        print()
    print("Mean all misses:")
    print(np.mean([
        simulated_misses[state] for state in correlation_dict.keys()
        if state in ('MI', 'PA', 'WI')
    ]))
    print("Std Dev all misses:")
    print(np.std([
        simulated_misses[state] for state in correlation_dict.keys()
        if state in ('MI', 'PA', 'WI')
    ]))


    # Calculate and print the correlation of the simulated misses
    simulated_correlation = np.corrcoef([simulated_misses[state] for state in correlation_dict.keys()])
    print("Simulated correlation matrix:")
    print(simulated_correlation)

    # Compare with the original correlation matrix
    original_correlation = np.array(
        [[correlation_dict[state1][state2] for state2 in correlation_dict.keys()] for state1 in
         correlation_dict.keys()])
    print("\nOriginal correlation matrix:")
    print(original_correlation)

    # Calculate the difference
    correlation_difference = np.abs(simulated_correlation - original_correlation)
    print("\nAbsolute difference between simulated and original correlation:")
    print(correlation_difference)
    print(f"Maximum difference: {np.max(correlation_difference)}")


def real_dist():
    dist, states = get_multivariate_t_dist(
        source='538',
        degrees_freedom=5,
    )
    samples = dist.rvs(size=100_000)
    print(samples)
    print(samples.shape)
    target = 0.02
    df = 5
    expected_scale = calc_scale_factor_for_t_dist(df, target)
    print("expected scale", expected_scale)
    print("Analytical?", (target / np.sqrt(2 / np.pi)) / np.sqrt(df / (df - 2)))
    print(np.abs(samples * expected_scale).mean(axis=0))


if __name__ == "__main__":
    real_dist()