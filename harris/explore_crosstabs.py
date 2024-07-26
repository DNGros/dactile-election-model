from pathlib import Path
import pandas as pd

from util import pd_all_columns, pd_all_rows

cur_path = (Path(__file__).parent).absolute()


def get_ecp_crosstabs() -> pd.DataFrame:
    # https://emersoncollegepolling.com/july-2024-national-poll-trump-46-biden-43/
    df = pd.read_excel(cur_path / "crosstabs" / "ecp_national_7.8.24.xlsx", sheet_name="crosstabs")
    with pd.option_context(*pd_all_columns, *pd_all_rows, 'display.max_colwidth', None):
        print(df)
    return df
    exit()


def main():
    df = get_ecp_crosstabs()


if __name__ == "__main__":
    main()