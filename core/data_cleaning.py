# Data Manipulation & Utilities
import numpy as np
import pandas as pd
import re

def column_summaries(df: pd.DataFrame) -> pd.DataFrame:
    summary_data = []
    for col_name in df.columns:
        col_dtype = df[col_name].dtype
        num_of_nulls = df[col_name].isnull().sum()
        num_of_non_nulls = df[col_name].notnull().sum()
        num_of_distinct_values = df[col_name].nunique()
        
        if num_of_distinct_values <= 10:
            distinct_values_counts = df[col_name].value_counts().to_dict()
        else:
            top_10_values_counts = df[col_name].value_counts().head(10).to_dict()
            distinct_values_counts = {k: v for k, v in sorted(top_10_values_counts.items(), key=lambda item: item[1], reverse=True)}

        summary_data.append({
            'col_name': col_name,
            'col_dtype': col_dtype,
            'num_of_nulls': num_of_nulls,
            'num_of_non_nulls': num_of_non_nulls,
            'num_of_distinct_values': num_of_distinct_values,
            'distinct_values_counts': distinct_values_counts
        })
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df

def missig_values_info(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum().reset_index(name='Counts')
    miss['Proportions (%)'] = miss['Counts']/len(df)*100
    return miss


def simplify_dtype(dtype: str):
    if dtype in (int, float, np.number): return 'Numeric'
    elif np.issubdtype(dtype, np.datetime64): return 'Datetime'
    elif dtype == str: return 'String'
    elif dtype == type(None): return 'Missing'
    else: return 'Other'

def analyze_column_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    all_dtypes = {'Numeric', 'Datetime', 'String', 'Missing', 'Other'}
    results = pd.DataFrame(index=df.columns, columns=list(all_dtypes), dtype=object).fillna('-')
    
    for column in df.columns:
        dtypes = df[column].apply(lambda x: simplify_dtype(type(x))).value_counts()
        percentages = (dtypes / len(df)) * 100
        for dtype, percent in percentages.items():
            if percent > 0:
                results.at[column, dtype] = f'{percent:.2f}%'  # Add % sign and format to 2 decimal places
            else:
                results.at[column, dtype] = '-'  # Add dash for 0%
    return results

def fit_transform(df: pd.DataFrame, cat_cols: list, rare_label: str = "Rare", threshold: float = 0.05) -> pd.DataFrame:
    
    df_transformed = df.copy()
    summary_records = []

    for col in cat_cols:
        value_counts = df[col].value_counts(dropna=False)
        total = len(df)

        if self.threshold < 1: rare_mask = (value_counts / total) < self.threshold
        else: rare_mask = value_counts < self.threshold
        rare_categories = value_counts[rare_mask].index.tolist()

        df_transformed[col] = df_transformed[col].apply(lambda x: self.rare_label if x in rare_categories else x)
        summary_records.append({
            "column": col,
            "unique_before": len(value_counts),
            "unique_after": df_transformed[col].nunique(),
            "rare_categories": rare_categories
        })

    summary_df = pd.DataFrame(summary_records)
    return df_transformed, summary_df

def interquartile_range_outlier(df: pd.DataFrame, numeric_cols: list =None, group: str = None) -> pd.DataFrame:
    if group: grouped = df.groupby(group)
    else: grouped = [(None, df)]
        
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if group in numeric_cols:
            numeric_cols.remove(group)
            

    results = []
    for group_name, group_df in grouped:    
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            count = df[(df[col] < lower) | (df[col] > upper)].shape[0]

            results.append({
                f'{group}': group_name,
                'Variable': col,
                'Counts': count,
                'Total Observations': len(group_df),
                'Proportion (%)': round((count / len(group_df)) * 100, 2),
            })
    results = pd.DataFrame(results)
    if group is None: results = results.drop(columns=[f'{group}'])        
    return results