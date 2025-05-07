import numpy as np
import pandas as pd




class DataProcessor():

    def __init__(self):
        # translation dicts: German -->> English
        self.metric_trs = {
            "Alkoholunfälle":     "Alcohol",
            "Fluchtunfälle":      "Hit-and-run",
            "Verkehrsunfälle":    "Traffic",
        }
        self.Accident_type_trs = {
            "insgesamt":                   "Total",
            "Verletzte und Getötete":      "Injuries and fatalities",
            "mit Personenschäden":         "With personal injuries",
        }

    def get_stats(self, df):
        """ returns general info on the input df """
        print(f"The number of the rows = {len(df)}")

        print("\n Dataframe information: ")
        print(df.info())

        print("\n Data summary statistics: ")
        return (df.describe())
    
    def processor(self, df):
        """ cleans and processes df """

        df = df[df['Value'].notna()] # we need no nan values in the target column

        # add Total Accident of each Year
        annual_totals = (
            df[(df['Accident_type'] == 'insgesamt') & (df['Month'] == 'Summe')]
            .groupby('Year')['Value']
            .sum()
            .rename('Total_Accidents_That_Year')
            .reset_index()
        )
        df = df.merge(annual_totals, on='Year', how='left')

        # Month column >> datetime format
        df['Month'] = pd.to_datetime(df['Month'].astype(str), format="%Y%m", errors='coerce') # 'coerce' removes "Summe" values and replases Nan 
        df['Month'] = df['Month'].dt.to_period('m')
        df = df[df['Month'].notna()]

        # get month number
        df['month_num'] = df['Month'].dt.month

        # cyclic encoding for seasonality
        df['sin_month'] = np.sin(2 * np.pi * df['month_num'] / 12)
        df['cos_month'] = np.cos(2 * np.pi * df['month_num'] / 12)

        # add season column
        def season(m):
            if m in (12,1,2): return 'Winter'
            if m in (3,4,5):  return 'Spring'
            if m in (6,7,8):  return 'Summer'
            return 'Autumn'
        df['Season'] = df['month_num'].apply(season)

        # categorical columns
        df['Accident_type'] = df['Accident_type'].astype('category')
        df['Category'] = df['Category'].astype('category')
        df['Season'] = df['Season'].astype('category')

        # translate 
        df['Category'] = df['Category'].replace(self.metric_trs)
        df['Accident_type'] = df['Accident_type'].replace(self.Accident_type_trs)

        # create the 1-month and 2_month lage features
        df = (df
            .sort_values(['Category','Accident_type','Month'])
            .groupby(['Category','Accident_type'])
            .apply(lambda g: g.assign(
                lag_1 = g['Value'].shift(1),
                lag_2 = g['Value'].shift(2)
            ))
            .reset_index(drop=True))
        
        return df 
    
    def drop_year(self, df, year=2020):
        """ removes any data that comes after [year] """
        
        # Validate year input
        if not isinstance(year, (int, float)):
            raise ValueError(f"Invalid year type: {type(year)}. Year must be a number.")
        if 'Year' not in df.columns:
            raise KeyError("DataFrame must contain a 'Year' column.")
        if year not in df['Year'].values:
            raise ValueError(f"The specified year {year} is not present in the DataFrame.")
        
        # Filter the DataFrame
        df_removed = df[df['Year'] > year]         
        df = df[df['Year'] <= year] 

        return df, df_removed
    
    def save_df(self, df, filename = "df_clean.pkl"):
        """ save clean dataframe """
        df.to_pickle(filename)
