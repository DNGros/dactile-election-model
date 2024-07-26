import pandas as pd
pd_all_columns = ('display.max_columns', None, 'display.width', None)
pd_all_rows = ('display.max_rows', None)

def get_only_value(x):
    if len(x) != 1:
        print(x)
        raise AssertionError(f"Expected only one value, got {len(x)}")
    return x[0]
