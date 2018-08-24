import pandas as pd
from pandas.compat import StringIO
from sklearn.metrics.classification import classification_report
import warnings
from functools import partial
from ml_recsys_tools.utils.parallelism import parallelize_dataframe


def console_settings():
    pd.set_option('display.max_colwidth', 300)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)


# def _split_json_field(df, field):
#     split_lambda = lambda x: pd.Series(json.loads(x) if x else [])
#     return pd.concat([df, df[field].astype(str).apply(split_lambda)], axis=1)

def _split_json_field(df, field):
    df_json = pd.read_json('[%s]' % ','.join(df[field].tolist()))
    return pd.concat([df.reset_index(), df_json], axis=1)


def split_json_field(df, field, remove_original=True, parallel=True):

    if parallel:
        df_out = parallelize_dataframe(df, partial(_split_json_field, field=field))
    else:
        df_out = _split_json_field(df, field=field)

    if remove_original:
        df_out.drop(field, axis=1, inplace=True)
    return df_out


def hist_by_groups(groups):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 7)
    ax = ax.ravel()
    for i, g in enumerate(groups):
        g[1].hist(label=g[0], alpha=1.0, ax=ax[i])
        ax[i].legend()
    return ax


def classification_report_df(y_true, y_pred):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return pd.read_csv(StringIO(classification_report(y_true, y_pred)), sep=r"[ \t]{2,}")


# slow
def explode_df(df, cols):
    static_cols = [col for col in df.columns if col not in cols]
    rows = []
    for _, row in df.iterrows():
        stat_list = row[static_cols].tolist()
        for i in range(len(row[cols[0]])):
            new_row = []
            for col in cols:
                new_row.append(row[col][i])
            new_row = new_row + stat_list
            rows.append(new_row)
    return pd.DataFrame(rows, columns=cols + static_cols)


def explode_df_parallel(df, cols):
    return parallelize_dataframe(df, partial(explode_df, cols=cols))

# def explode_df_np(df, cols):
#     static_cols = [col for col in df.columns if col not in cols]
#     new_mats = []
#     for _, row in df.iterrows():
#         stat_arr = np.matrix(row[static_cols].tolist())
#         n_rep = len(row[cols[0]])
#         stat_mat = np.repeat(stat_arr, n_rep, axis=0)
#         rows_mat = np.concatenate((stat_mat, *tuple(np.matrix(row[col]).T for col in cols)), axis=1)
#         new_mats.append(rows_mat)
#
#     full_mat = np.concatenate(tuple(new_mats), axis=0)
#     return pd.DataFrame(full_mat, columns=static_cols+cols)


# def explode_df(df, cols, fill_value=''):
#     '''
#     from here:
#         https://stackoverflow.com/questions/45846765/efficient-way-to-unnest-explode-multiple-list-columns-in-a-pandas-dataframe
#     :param df:
#     :param cols:
#     :param fill_value:
#     :return:
#     '''
#     # make sure `lst_cols` is a list
#     if cols and not isinstance(cols, list):
#         cols = [cols]
#     # all columns except `lst_cols`
#     idx_cols = df.columns.difference(cols)
#
#     # calculate lengths of lists
#     lens = df[cols[0]].str.len()
#
#     if (lens > 0).all():
#         # ALL lists in cells aren't empty
#         return pd.DataFrame({
#             col:np.repeat(df[col].values, df[cols[0]].str.len())
#             for col in idx_cols
#         }).assign(**{col:np.concatenate(df[col].values) for col in cols}) \
#           .loc[:, df.columns]
#     else:
#         # at least one list in cells is empty
#         return pd.DataFrame({
#             col:np.repeat(df[col].values, df[cols[0]].str.len())
#             for col in idx_cols
#         }).assign(**{col:np.concatenate(df[col].values) for col in cols}) \
#           .append(df.loc[lens==0, idx_cols]).fillna(fill_value) \
#           .loc[:, df.columns]
