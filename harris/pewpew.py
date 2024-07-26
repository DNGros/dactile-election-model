from pathlib import Path

import pandas as pd


if __name__ == "__main__":
    df = pd.read_spss(Path("~/Downloads/W116_Oct22/ATP W116.sav").expanduser())
    print(df)