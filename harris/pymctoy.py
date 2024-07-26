import pymc as pm
import numpy as np
import matplotlib.pyplot as plt

def main():
    import pymc as pm
    import numpy as np
    import matplotlib.pyplot as plt

    # Set the random seed for reproducibility
    np.random.seed(0)

    # Define the number of voters
    n_voters = 1000

    with pm.Model() as voter_model:
        # Probabilities for each voter type
        p_voter_types = pm.Dirichlet('p_voter_types', a=[1, 1, 1])

        # Voter type (0 for identity, 1 for issue, 2 for headlines)
        voter_type = pm.Categorical('voter_type', p=p_voter_types, shape=n_voters)

        # Probability of voting Democrat for each voter type
        p_dem_identity = 0.9
        p_dem_issue = 0.2
        p_dem_headlines = 0.3  # You can adjust this value

        # Voting choice (0 for Republican, 1 for Democrat)
        vote = pm.Bernoulli('vote',
                            p=pm.math.switch(voter_type,
                                             p_dem_identity,
                                             pm.math.switch(voter_type - 1,
                                                            p_dem_issue,
                                                            p_dem_headlines)),
                            shape=n_voters)

        # Generate synthetic data
        trace = pm.sample_prior_predictive(samples=1)

        # Visualize the model
        plt.figure(figsize=(12, 10))
        pm.model_to_graphviz(voter_model).render("voter_model_graph", format="png", cleanup=True)
        plt.imshow(plt.imread("../voter_model_graph.png"))
        plt.axis('off')
        plt.title("Voter Model Graph with Three Voter Types")
        plt.show()

    # Calculate and print the results
    votes = trace.prior.vote[0]
    voter_types = trace.prior.voter_type[0]
    dem_votes = votes.sum().item()
    rep_votes = n_voters - dem_votes

    print(f"Democratic votes: {dem_votes}")
    print(f"Republican votes: {rep_votes}")
    print(f"Democratic vote share: {dem_votes / n_voters:.2%}")

    # Count voter types
    identity_voters = np.sum(voter_types == 0)
    issue_voters = np.sum(voter_types == 1)
    headline_voters = np.sum(voter_types == 2)


if __name__ == "__main__de":
    main()
