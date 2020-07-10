import pandas as pd
from os import path
import numpy as np
import h5py

def function_name(argument):
    print('example')

if __name__ =="__main__":
    # remember indentations!
    filename = 'data/ethnic fractionalization - original'
    excel_file = filename +'.xlsx'
    h5_file = filename + '.h5'

    output_filename = 'results/HIEF_dataset_v2.xlsx'

    dataset=[]
    if not path.exists(h5_file):
        print('Reading dataset from excel file...')
        dataset = pd.read_excel(excel_file)
        print('Saving dataset to h5 format...')
        dataset.to_hdf(h5_file, key='df', mode='w')
    else:
        print('Reading dataset from h5 file...')
        dataset = pd.read_hdf(h5_file, 'df')

    # this will contain the resulting ef_indexes
    HIEF_data = [] # create a list and then append it to an empty dataframe, for cheaper computation

    countries = dataset['Country'].unique()
    for country in countries:
        print(country)
        data_country = dataset.loc[(dataset['Country'] == country)]
        years = data_country['Year'].unique()
        #print (years)
        for year in years:
            subset = data_country.loc[(data_country['Year'] == year) ]
            #print('subset', subset['Group Estimate'])
            cleaned_subset = subset.drop_duplicates(subset='Group Estimate', keep="first")
            #print('cleaned ', cleaned_subset['Group Estimate'])
            samples = cleaned_subset['Group Estimate']
            ef_index = 1 - np.sum ( np.power( samples / 100, 2) )
            if ef_index < 1:
                HIEF_data.append( [ country, year, ef_index] )

    HIEF_dataframe= pd.DataFrame(HIEF_data, columns=['Country', 'Year', 'EF_index'])
    HIEF_dataframe.to_excel(output_filename, index=False)
    #print(dataset.head())
    #print(dataset.tail())
    #print(dataset.info())
    #print('shape ', dataset.shape)