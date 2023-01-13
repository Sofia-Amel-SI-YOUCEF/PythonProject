import pandas as pd



def get_features(df, corr_thresh=.02, corr_df=False):
    corr_list = []
    for col in df.columns:
        corr_list.append(round(df["TARGET"].corr(df[col]), 2))
    df_corr = pd.DataFrame(data=zip(df.columns.tolist(), corr_list),
                           columns=["col_name", "corr"]) \
        .sort_values("corr", ascending=False) \
        .reset_index(drop=True)
    df_corr = df_corr[abs(df_corr["corr"]) > corr_thresh][1:].reset_index(drop=True)
    features = df_corr["col_name"].tolist()
    if not corr_df:
        return features
    return features, df_corr


def select_features(exp):
    """
    :param exp: explain_instance obj
        local feature importance computed for a data instance
    :return: list
     First (10) important features selected by explain_instance
    """
    s_features = [feature.split("<")[0] for feature, value in exp.as_list()]
    s_features = [feature.split(">")[0].strip() for feature in s_features]
    return s_features
