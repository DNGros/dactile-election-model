from state_correlations import load_five_thirty_eight_correlation, load_economist_correlations


def main():
    c538 = load_five_thirty_eight_correlation()['MI']
    cEcon = load_economist_correlations()['MI']
    vals = [
        "WI",
        "NE",
        "PA",
        "MI",
        "NV",
        "GA",
        "AZ",
        "NC",
    ]
    for v in vals:
        print(f"{(c538[v] + cEcon[v])/2:.2f} {{{c538[v]:.2f}, {cEcon[v]:.2f}}}")
    power = 3
    print(f"--- pow {power}")
    for v in vals:
        print(f"{(c538[v]**power + cEcon[v]**power)/2:.2f} {{{c538[v]**power:.2f}, {cEcon[v]**power:.2f}}}")


if __name__ == "__main__":
    main()