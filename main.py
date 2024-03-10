import dill
from datetime import datetime

from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

def filter_data(df):
    df = df.copy()
    df = df.drop(['rn', 'pre_loans_total_overdue'], axis=1)


    df['enc_paym_count'] = (df['enc_paym_0'] + df['enc_paym_1'] + df['enc_paym_2'] + df['enc_paym_3']
                            + df['enc_paym_4']
                            + df['enc_paym_5'] + df['enc_paym_6'] + df['enc_paym_7'] + df['enc_paym_8']
                            + df['enc_paym_9'] + df['enc_paym_10'] + df['enc_paym_11'] + df['enc_paym_12']
                            + df['enc_paym_13'] + df['enc_paym_14'] +
                            df['enc_paym_15'] + df['enc_paym_16'] + df['enc_paym_17'] +
                            df['enc_paym_18'] + df['enc_paym_19'] + df['enc_paym_20'] + df['enc_paym_21'] +
                            df['enc_paym_22'] + df['enc_paym_23'] + df['enc_paym_24'])

    df = df.drop(
        ['enc_paym_0', 'enc_paym_1', 'enc_paym_2', 'enc_paym_3', 'enc_paym_4', 'enc_paym_5', 'enc_paym_6', 'enc_paym_7',
         'enc_paym_8', 'enc_paym_9', 'enc_paym_10', 'enc_paym_11', 'enc_paym_12', 'enc_paym_13', 'enc_paym_14',
         'enc_paym_15', 'enc_paym_16', 'enc_paym_17', 'enc_paym_18', 'enc_paym_19', 'enc_paym_20', 'enc_paym_21',
         'enc_paym_22', 'enc_paym_23', 'enc_paym_24'], axis=1)

    return df

def clear(df):
    df = df.copy()
    agg_func_transform = {
        'pre_loans_max_overdue_sum': ['mean'],

        'pre_loans5': ['mean'], 'pre_loans530': ['mean'], 'pre_loans3060': ['mean'],
        'pre_loans6090': ['mean'], 'pre_loans90': ['mean'],

        'enc_paym_count': ['mean', 'sum'],

        'is_zero_loans5': ['mean'], 'is_zero_loans530': ['mean'], 'is_zero_loans3060': ['mean'],
        'is_zero_loans6090': ['mean'], 'is_zero_loans90': ['mean'],

        'is_zero_util_0': ['mean', 'median', 'sum'], 'is_zero_util_1': ['mean', 'median', 'sum'],
        'is_zero_over2limit_0': ['mean', 'median'], 'is_zero_over2limit_1': ['mean', 'median'],
        'is_zero_maxover2limit_0': ['mean', 'median'], 'is_zero_maxover2limit_1': ['mean', 'median'],
        'enc_loans_account_holder_type_0': ['mean', 'median'], 'enc_loans_account_holder_type_1': ['mean', 'median'],
        'enc_loans_account_holder_type_2': ['mean', 'median'], 'enc_loans_account_holder_type_3': ['mean', 'median'],
        'enc_loans_account_holder_type_4': ['mean', 'median'], 'enc_loans_account_holder_type_5': ['mean', 'median'],
        'enc_loans_account_holder_type_6': ['mean', 'median'], 'enc_loans_credit_status_0': ['mean', 'median'],
        'enc_loans_credit_status_1': ['mean', 'median'], 'enc_loans_credit_status_2': ['mean', 'median'],
        'enc_loans_credit_status_3': ['mean', 'median'], 'enc_loans_credit_status_4': ['mean', 'median'],
        'enc_loans_credit_status_5': ['mean', 'median'], 'enc_loans_credit_status_6': ['mean', 'median'],

        'enc_loans_account_cur_0': ['mean'], 'enc_loans_account_cur_1': ['mean'],
        'enc_loans_account_cur_2': ['mean'], 'enc_loans_account_cur_3': ['mean'],
        'enc_loans_credit_type_0': ['mean'], 'enc_loans_credit_type_1': ['mean'],
        'enc_loans_credit_type_2': ['mean'], 'enc_loans_credit_type_3': ['mean'],
        'enc_loans_credit_type_4': ['mean'], 'enc_loans_credit_type_5': ['mean'],

        'pclose_flag_0': ['mean'],
        'pclose_flag_1': ['mean'], 'fclose_flag_0': ['mean'], 'fclose_flag_1': ['mean'],

        'pre_since_opened_0': ['mean', 'max', 'sum'], 'pre_since_opened_1': ['mean', 'max', 'sum'],
        'pre_since_opened_2': ['mean', 'max', 'sum'],
        'pre_since_opened_3': ['mean', 'max', 'sum'], 'pre_since_opened_4': ['mean', 'max', 'sum'],
        'pre_since_opened_5': ['mean', 'max', 'sum'],
        'pre_since_opened_6': ['mean', 'max', 'sum'], 'pre_since_opened_7': ['mean', 'max', 'sum'],
        'pre_since_opened_8': ['mean', 'max', 'sum'],
        'pre_since_opened_9': ['mean', 'max', 'sum'], 'pre_since_opened_10': ['mean', 'max', 'sum'],
        'pre_since_opened_11': ['mean', 'max', 'sum'],
        'pre_since_opened_12': ['mean', 'max', 'sum'], 'pre_since_opened_13': ['mean', 'max', 'sum'],
        'pre_since_opened_14': ['mean', 'max', 'sum'],
        'pre_since_opened_15': ['mean', 'max', 'sum'], 'pre_since_opened_16': ['mean', 'max', 'sum'],
        'pre_since_opened_17': ['mean', 'max', 'sum'],
        'pre_since_opened_18': ['mean', 'max', 'sum'], 'pre_since_opened_19': ['mean', 'max', 'sum'],
        'pre_since_confirmed_0': ['mean', 'max', 'sum'],
        'pre_since_confirmed_1': ['mean', 'max', 'sum'], 'pre_since_confirmed_2': ['mean', 'max', 'sum'],
        'pre_since_confirmed_3': ['mean', 'max', 'sum'], 'pre_since_confirmed_4': ['mean', 'max', 'sum'],
        'pre_since_confirmed_5': ['mean', 'max', 'sum'], 'pre_since_confirmed_6': ['mean', 'max', 'sum'],
        'pre_since_confirmed_7': ['mean', 'max', 'sum'], 'pre_since_confirmed_8': ['mean', 'max', 'sum'],
        'pre_since_confirmed_9': ['mean', 'max', 'sum'], 'pre_since_confirmed_10': ['mean', 'max', 'sum'],
        'pre_since_confirmed_11': ['mean', 'max', 'sum'], 'pre_since_confirmed_12': ['mean', 'max', 'sum'],
        'pre_since_confirmed_13': ['mean', 'max', 'sum'], 'pre_since_confirmed_14': ['mean', 'max', 'sum'],
        'pre_since_confirmed_15': ['mean', 'max', 'sum'], 'pre_since_confirmed_16': ['mean', 'max', 'sum'],
        'pre_since_confirmed_17': ['mean', 'max', 'sum'],

        'pre_pterm_0': ['mean', 'max', 'sum', 'median'],
        'pre_pterm_1': ['mean', 'max', 'sum', 'median'], 'pre_pterm_2': ['mean', 'max', 'sum', 'median'],
        'pre_pterm_3': ['mean', 'max', 'sum', 'median'], 'pre_pterm_4': ['mean', 'max', 'sum', 'median'],
        'pre_pterm_5': ['mean', 'max', 'sum', 'median'], 'pre_pterm_6': ['mean', 'max', 'sum', 'median'],
        'pre_pterm_7': ['mean', 'max', 'sum', 'median'], 'pre_pterm_8': ['mean', 'max', 'sum', 'median'],
        'pre_pterm_9': ['mean', 'max', 'sum', 'median'], 'pre_pterm_10': ['mean', 'max', 'sum', 'median'],
        'pre_pterm_11': ['mean', 'max', 'sum', 'median'], 'pre_pterm_12': ['mean', 'max', 'sum', 'median'],
        'pre_pterm_13': ['mean', 'max', 'sum', 'median'], 'pre_pterm_14': ['mean', 'max', 'sum', 'median'],
        'pre_pterm_15': ['mean', 'max', 'sum', 'median'], 'pre_pterm_16': ['mean', 'max', 'sum', 'median'],
        'pre_pterm_17': ['mean', 'max', 'sum', 'median'], 'pre_fterm_0': ['mean', 'max', 'sum', 'median'],
        'pre_fterm_1': ['mean', 'max', 'sum', 'median'], 'pre_fterm_2': ['mean', 'max', 'sum', 'median'],
        'pre_fterm_3': ['mean', 'max', 'sum', 'median'], 'pre_fterm_4': ['mean', 'max', 'sum', 'median'],
        'pre_fterm_5': ['mean', 'max', 'sum', 'median'], 'pre_fterm_6': ['mean', 'max', 'sum', 'median'],
        'pre_fterm_7': ['mean', 'max', 'sum', 'median'], 'pre_fterm_8': ['mean', 'max', 'sum', 'median'],
        'pre_fterm_9': ['mean', 'max', 'sum', 'median'], 'pre_fterm_10': ['mean', 'max', 'sum', 'median'],
        'pre_fterm_11': ['mean', 'max', 'sum', 'median'], 'pre_fterm_12': ['mean', 'max', 'sum', 'median'],
        'pre_fterm_13': ['mean', 'max', 'sum', 'median'], 'pre_fterm_14': ['mean', 'max', 'sum', 'median'],
        'pre_fterm_15': ['mean', 'max', 'sum', 'median'], 'pre_fterm_16': ['mean', 'max', 'sum', 'median'],

        'pre_till_pclose_0': ['mean', 'sum', 'min'], 'pre_till_pclose_1': ['mean', 'sum', 'min'],
        'pre_till_pclose_2': ['mean', 'sum', 'min'], 'pre_till_pclose_3': ['mean', 'sum', 'min'],
        'pre_till_pclose_4': ['mean', 'sum', 'min'],
        'pre_till_pclose_5': ['mean', 'sum', 'min'], 'pre_till_pclose_6': ['mean', 'sum', 'min'],
        'pre_till_pclose_7': ['mean', 'sum', 'min'],
        'pre_till_pclose_8': ['mean', 'sum', 'min'], 'pre_till_pclose_9': ['mean', 'sum', 'min'],
        'pre_till_pclose_10': ['mean', 'sum', 'min'],
        'pre_till_pclose_11': ['mean', 'sum', 'min'], 'pre_till_pclose_12': ['mean', 'sum', 'min'],
        'pre_till_pclose_13': ['mean', 'sum', 'min'],
        'pre_till_pclose_14': ['mean', 'sum', 'min'], 'pre_till_pclose_15': ['mean', 'sum', 'min'],
        'pre_till_pclose_16': ['mean', 'sum', 'min'],
        'pre_till_fclose_0': ['mean', 'sum', 'min'], 'pre_till_fclose_1': ['mean', 'sum', 'min'],
        'pre_till_fclose_2': ['mean', 'sum', 'min'],
        'pre_till_fclose_3': ['mean', 'sum', 'min'], 'pre_till_fclose_4': ['mean', 'sum', 'min'],
        'pre_till_fclose_5': ['mean', 'sum', 'min'],
        'pre_till_fclose_6': ['mean', 'sum', 'min'], 'pre_till_fclose_7': ['mean', 'sum', 'min'],
        'pre_till_fclose_8': ['mean', 'sum', 'min'],
        'pre_till_fclose_9': ['mean', 'sum', 'min'], 'pre_till_fclose_10': ['mean', 'sum', 'min'],
        'pre_till_fclose_11': ['mean', 'sum', 'min'],
        'pre_till_fclose_12': ['mean', 'sum', 'min'], 'pre_till_fclose_13': ['mean', 'sum', 'min'],
        'pre_till_fclose_14': ['mean', 'sum', 'min'],
        'pre_till_fclose_15': ['mean', 'sum', 'min'],

        'pre_loans_next_pay_summ_0': ['mean', 'sum'],
        'pre_loans_next_pay_summ_1': ['mean', 'sum'], 'pre_loans_next_pay_summ_2': ['mean', 'sum'],
        'pre_loans_next_pay_summ_3': ['mean', 'sum'], 'pre_loans_next_pay_summ_4': ['mean', 'sum'],
        'pre_loans_next_pay_summ_5': ['mean', 'sum'], 'pre_loans_next_pay_summ_6': ['mean', 'sum'],

        'pre_loans_outstanding_1': ['mean', 'sum', 'min', 'first', 'last'],
        'pre_loans_outstanding_2': ['mean', 'sum', 'min', 'first', 'last'],
        'pre_loans_outstanding_3': ['mean', 'sum', 'min', 'first', 'last'],
        'pre_loans_outstanding_4': ['mean', 'sum', 'min', 'first', 'last'],
        'pre_loans_outstanding_5': ['mean', 'sum', 'min', 'first', 'last'],

        'pre_maxover2limit_0': ['mean'], 'pre_maxover2limit_1': ['mean'], 'pre_maxover2limit_2': ['mean'],
        'pre_maxover2limit_3': ['mean'], 'pre_maxover2limit_4': ['mean'], 'pre_maxover2limit_5': ['mean'],
        'pre_maxover2limit_6': ['mean'], 'pre_maxover2limit_7': ['mean'], 'pre_maxover2limit_8': ['mean'],
        'pre_maxover2limit_9': ['mean'], 'pre_maxover2limit_10': ['mean'], 'pre_maxover2limit_11': ['mean'],
        'pre_maxover2limit_12': ['mean'], 'pre_maxover2limit_13': ['mean'], 'pre_maxover2limit_14': ['mean'],
        'pre_maxover2limit_15': ['mean'], 'pre_maxover2limit_16': ['mean'], 'pre_maxover2limit_17': ['mean'],
        'pre_maxover2limit_18': ['mean'], 'pre_maxover2limit_19': ['mean'],

        'pre_util_0': ['mean'], 'pre_util_1': ['mean'], 'pre_util_2': ['mean'], 'pre_util_3': ['mean'],
        'pre_util_4': ['mean'],
        'pre_util_5': ['mean'], 'pre_util_6': ['mean'], 'pre_util_7': ['mean'], 'pre_util_8': ['mean'],
        'pre_util_9': ['mean'],
        'pre_util_10': ['mean'], 'pre_util_11': ['mean'], 'pre_util_12': ['mean'], 'pre_util_13': ['mean'],
        'pre_util_14': ['mean'], 'pre_util_15': ['mean'], 'pre_util_16': ['mean'], 'pre_util_17': ['mean'],
        'pre_util_18': ['mean'], 'pre_util_19': ['mean'],

        'pre_loans_credit_limit_0': ['mean', 'sum'], 'pre_loans_credit_limit_1': ['mean', 'sum'],
        'pre_loans_credit_limit_2': ['mean', 'sum'], 'pre_loans_credit_limit_3': ['mean', 'sum'],
        'pre_loans_credit_limit_4': ['mean', 'sum'], 'pre_loans_credit_limit_5': ['mean', 'sum'],
        'pre_loans_credit_limit_6': ['mean', 'sum'], 'pre_loans_credit_limit_7': ['mean', 'sum'],
        'pre_loans_credit_limit_8': ['mean', 'sum'], 'pre_loans_credit_limit_9': ['mean', 'sum'],
        'pre_loans_credit_limit_10': ['mean', 'sum'], 'pre_loans_credit_limit_11': ['mean', 'sum'],
        'pre_loans_credit_limit_12': ['mean', 'sum'], 'pre_loans_credit_limit_13': ['mean', 'sum'],
        'pre_loans_credit_limit_14': ['mean', 'sum'], 'pre_loans_credit_limit_15': ['mean', 'sum'],
        'pre_loans_credit_limit_16': ['mean', 'sum'], 'pre_loans_credit_limit_17': ['mean', 'sum'],
        'pre_loans_credit_limit_18': ['mean', 'sum'], 'pre_loans_credit_limit_19': ['mean', 'sum'],

        'pre_loans_credit_cost_rate_0': ['median'],
        'pre_loans_credit_cost_rate_1': ['median'], 'pre_loans_credit_cost_rate_2': ['median'],
        'pre_loans_credit_cost_rate_3': ['median'], 'pre_loans_credit_cost_rate_4': ['median'],
        'pre_loans_credit_cost_rate_5': ['median'], 'pre_loans_credit_cost_rate_6': ['median'],
        'pre_loans_credit_cost_rate_7': ['median'], 'pre_loans_credit_cost_rate_8': ['median'],
        'pre_loans_credit_cost_rate_9': ['median'], 'pre_loans_credit_cost_rate_10': ['median'],
        'pre_loans_credit_cost_rate_11': ['median'], 'pre_loans_credit_cost_rate_12': ['median'],
        'pre_loans_credit_cost_rate_13': ['median'],

        'pre_over2limit_0': ['mean', 'median', 'sum', 'max', 'first', 'last'],
        'pre_over2limit_1': ['mean', 'median', 'sum', 'max', 'first', 'last'],
        'pre_over2limit_2': ['mean', 'median', 'sum', 'max', 'first', 'last'],
        'pre_over2limit_3': ['mean', 'median', 'sum', 'max', 'first', 'last'],
        'pre_over2limit_4': ['mean', 'median', 'sum', 'max', 'first', 'last'],
        'pre_over2limit_5': ['mean', 'median', 'sum', 'max', 'first', 'last'],
        'pre_over2limit_6': ['mean', 'median', 'sum', 'max', 'first', 'last'],
        'pre_over2limit_7': ['mean', 'median', 'sum', 'max', 'first', 'last'],
        'pre_over2limit_8': ['mean', 'median', 'sum', 'max', 'first', 'last'],
        'pre_over2limit_9': ['mean', 'median', 'sum', 'max', 'first', 'last'],
        'pre_over2limit_10': ['mean', 'median', 'sum', 'max', 'first', 'last'],
        'pre_over2limit_11': ['mean', 'median', 'sum', 'max', 'first', 'last'],
        'pre_over2limit_12': ['mean', 'median', 'sum', 'max', 'first', 'last'],
        'pre_over2limit_13': ['mean', 'median', 'sum', 'max', 'first', 'last'],
        'pre_over2limit_14': ['mean', 'median', 'sum', 'max', 'first', 'last'],
        'pre_over2limit_15': ['mean', 'median', 'sum', 'max', 'first', 'last'],
        'pre_over2limit_16': ['mean', 'median', 'sum', 'max', 'first', 'last'],
        'pre_over2limit_17': ['mean', 'median', 'sum', 'max', 'first', 'last'],
        'pre_over2limit_18': ['mean', 'median', 'sum', 'max', 'first', 'last'],
        'pre_over2limit_19': ['mean', 'median', 'sum', 'max', 'first', 'last'],

    }
    df = df.groupby("id").agg(agg_func_transform).round(2)
    df = df.drop_duplicates()
    return df

def filter_data2(df):
    df = df.copy()
    col = df.columns

    def calculate_boundaries(series):
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr = q75 - q25

        boundaries = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)
        return boundaries

    for d in [0, 5, 50, 36, 39, 41, 44, 102, 129, 151]:
        boundaries_max = calculate_boundaries(df[col[d]])
        is_outlier = (df[col[d]] > int(boundaries_max[1]))
        df.loc[is_outlier, [col[d]]] = int(boundaries_max[1])

    for s in [2, 3]:
        boundaries_max = calculate_boundaries(df[col[s]])
        is_outlier = (df[col[s]] < int(boundaries_max[0]))
        df.loc[is_outlier, [col[s]]] = int(boundaries_max[0])
    return df



def main():
    categorical_features = ['is_zero_util',
                            'pre_since_opened', 'pre_since_confirmed',
                            'is_zero_over2limit', 'is_zero_maxover2limit', 'enc_loans_account_holder_type',
                            'enc_loans_credit_status', 'enc_loans_account_cur',
                            'enc_loans_credit_type', 'pclose_flag', 'fclose_flag',
                            'pre_pterm', 'pre_fterm',
                            'pre_till_pclose', 'pre_till_fclose',
                            'pre_loans_next_pay_summ',
                            'pre_loans_outstanding',
                            'pre_maxover2limit',
                            'pre_util',
                            'pre_loans_credit_limit',
                            'pre_loans_credit_cost_rate',
                            'pre_over2limit',
                            ]

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    column_transformer = ColumnTransformer(transformers=[
        ('categorical', categorical_transformer, categorical_features)
    ])

    preprocessor = Pipeline(steps=[
        ('filter', FunctionTransformer(filter_data)),
        ('column_transformer', column_transformer),
        ('clear', FunctionTransformer(clear)),
        ('filter2', FunctionTransformer(filter_data2)),
    ])

    model = joblib.load('model_for_pipline.pkl')


    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])


    with open('bank_pipe.pkl', 'wb') as file:
        dill.dump({
            'model': pipe,
            'metadata': {
                'name': 'bank model',
                'author': 'Andrei',
                'version': 1,
                'date': datetime.now(),
                'type': 'logistic regression',
                'accuracy': 0.75
            }
        }, file)


if __name__ == '__main__':
    main()