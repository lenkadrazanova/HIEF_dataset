# This script creates the HIEF dataset published in
# Dražanová, Lenka (2019). Historical Index of Ethnic Fractionalization Dataset (HIEF), Harvard Dataverse, V1.0, https://doi.org/10.7910/DVN/4JQRCL
#
# Data robustness checks is performed by creating 2 additional datasets with noise. Adapted from:
# Kolo, Philipp (2012) : Measuring a new aspect of ethnicity: The appropriate
# diversity index, IAI Discussion Papers, No. 221, Georg-August-Universität Göttingen, Ibero-
# America Institute for Economic Research (IAI), Göttingen
import pandas as pd
from os import path
import numpy as np
import h5py
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
from scipy import stats


## if noise_type = 0, no noise is added to the group estimate
# otherwise, a random noise sampled from a normal distribution is added to each group estimate.
# In particular:
## if noise_type = 1, the standard deviation of the Gaussian noise is set to the standard deviation
# of the group distribution over all observations (thus equal for all countries)
## if noise_type = 2, the standard deviation of the Gaussian noise is set to a
# country specific standard deviation
## if noise_type = 3, remove from each country/year the smallest group (if nr. of groups >1 and
# group size < 1 percent)
def create_hief_dataset(filename, output_filename, noise_type = 0):
    dataset=[]
    # use h5 format for faster reading from file.
    # If dataset.h5 does not exists, reads from excel file and saves it into h5
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

    # calculate the std dev over all the observations (sigma_1)
    if noise_type ==1:
        samp_std = dataset.loc[:,'Group Estimate'] / 100
        sigma_1 = samp_std.std(skipna = True)
    # get the list of countries
    countries = dataset['Country'].unique()
    # iterate over each country
    for country in countries:
        print(country)
        # get the data for this country
        data_country = dataset.loc[(dataset['Country'] == country)]
        # calculate the country specific standard deviation (sigma_2)
        if noise_type == 2:
            samp_std = dataset['Group Estimate'].loc[(dataset['Country'] == country)] / 100
            sigma_2 = samp_std.std(skipna=True)

        # get the list of the years for this country
        years = data_country['Year'].unique()
        # iterate over these years
        for year in years:
            subset = data_country.loc[(data_country['Year'] == year) ]
            #print('subset', subset['Group Estimate'])
            cleaned_subset = subset.drop_duplicates(subset='Group Estimate', keep="first")
            #print('cleaned ', cleaned_subset['Group Estimate'])
            samples = cleaned_subset['Group Estimate']
            samples_sum=0 # keep the sum before eventually removing groups
            # add noise, if requested
            if noise_type == 1:
                samples = samples * (1 + np.random.normal(0, sigma_1, len(samples)) ) #  np.random.normal(mu, sigma)
                samples = np.fabs(samples) # prevents having negative numbers
                samples_sum = samples.sum()
            if noise_type == 2:
                samples = samples * (1 + np.random.normal(0, sigma_2, len(samples)) ) #  np.random.normal(mu, sigma)
                samples = np.fabs(samples) # prevents having negative numbers
                samples_sum = samples.sum()
            if noise_type == 3: # remove smallest group for each country/year
                print(samples)
                samples_sum = samples.sum()
                if len(samples)>1 and  samples.values.argmin() < 1:
                    samples = np.delete(samples.values, samples.values.argmin())
                print('smaller: ', samples)

            # calculate the index
            ef_index = 1 - np.sum ( np.power( samples / samples_sum, 2) )
            if 0 <= ef_index < 1: # makes sure to append only valid values
                HIEF_data.append( [ country, year, ef_index] )

    # create the dataframe with the collected list of HIEFs
    HIEF_dataframe= pd.DataFrame(HIEF_data, columns=['Country', 'Year', 'EF_index'])
    # save it
    HIEF_dataframe.to_excel(output_filename, index=False)
    # returns it as a numpy array, for plots and statistics
    return np.asarray(HIEF_data)

# Create plots and makes statistics
# Original: the HIEF dataset created with the original dataset
# sigma_1: the HIEF dataset created with the original dataset with added gaussian noise (standard
# deviation set to the stddev of the group distribution over all observations (thus equal for all countries)
# sigma_2: the HIEF dataset created with the original dataset with added gaussian noise (standard
# deviation set to the country specific standard deviation)
# smaller: the HIEF dataset created with the original dataset removed with tha smallest group (from
# each country/year, only if nr. of groups >1 and if group size < 1 percent
# Shape of each dataset is nr_samples x (country, year, EF_index)
def do_plots(original, sigma_1, sigma_2, smaller):
    print('Original HIEF dataset shape ', np.asarray(original).shape)
    print('Sigma_1 HIEF dataset shape ', np.asarray(sigma_1).shape)
    print('Sigma_2 HIEF dataset shape ', np.asarray(sigma_2).shape)
    print('Smaller HIEF dataset shape ', np.asarray(smaller).shape)

    # calculate Pearson's correlation
    corr_1, _ = pearsonr(original[:,2], sigma_1[:,2])
    print('Pearsons correlation between original and sigma_1: %.3f' % corr_1)
    # calculate Pearson's correlation
    corr_2, _ = pearsonr(original[:,2], sigma_2[:,2])
    print('Pearsons correlation between original and sigma_2: %.3f' % corr_2)
    # calculate Pearson's correlation
    corr_3, _ = pearsonr(original[:,2], smaller[:,2])
    print('Pearsons correlation between original and smaller: %.3f' % corr_3)

    # make plot for sigma_1
    fig = plt.figure(1, figsize=(6, 6))
    plt.scatter(original[:,2], sigma_1[:,2], s=2, alpha=0.3)
    plt.xlabel('HIEF original')
    plt.ylabel('HIEF sigma_1')
    plt.savefig('results/sigma_1.png')
    plt.close(fig)

    # make plot for sigma 2
    fig = plt.figure(1, figsize=(6, 6))
    plt.scatter(original[:,2], sigma_2[:,2], s=2, alpha=0.3)
    plt.xlabel('HIEF original')
    plt.ylabel('HIEF sigma_2')
    plt.savefig('results/sigma_2.png')
    plt.close(fig)

    # make plot for smaller
    fig = plt.figure(1, figsize=(6, 6))
    plt.scatter(original[:,2], smaller[:,2], s=2, alpha=0.3)
    plt.xlabel('HIEF original')
    plt.ylabel('HIEF smaller')
    plt.savefig('results/smaller.png')
    plt.close(fig)


    # t-tests
    res1 = stats.ttest_ind(original[:,2], sigma_1[:,2])
    print ('Paired t-test original vs sigma1: ', res1)

    res2 = stats.ttest_ind(original[:,2], sigma_2[:,2])
    print ('Paired t-test original vs sigma2: ', res2)

    res3 = stats.ttest_ind(original[:,2], smaller[:,2])
    print ('Paired t-test original vs smaller: ', res3)

if __name__ =="__main__":
    # remember indentations!
    filename = 'data/ethnic fractionalization - original'
    excel_file = filename +'.xlsx'
    h5_file = filename + '.h5'

    output_filename_0 = 'results/HIEF_dataset_v2.xlsx'
    output_filename_1 = 'results/HIEF_dataset_v2_sigma1.xlsx'
    output_filename_2 = 'results/HIEF_dataset_v2_sigma2.xlsx'
    output_filename_3 = 'results/HIEF_dataset_v2_smaller.xlsx'

    print ('Creating HIEF original')
    if not path.exists(output_filename_0):
        hief_original = create_hief_dataset(filename=filename, output_filename=output_filename_0, noise_type=0)
    else:
        print('File ',output_filename_0, ' already exists')
        hief_original = pd.read_excel(output_filename_0).to_numpy()

    print ('Creating HIEF with noise type 1, sigma1')
    if not path.exists(output_filename_1):
        hief_sigma1 = create_hief_dataset(filename=filename, output_filename=output_filename_1, noise_type=1)
    else:
        print('File ',output_filename_1, ' already exists')
        hief_sigma1 = pd.read_excel(output_filename_1).to_numpy()

    print ('Creating HIEF with noise type 2, sigma2')
    if not path.exists(output_filename_2):
        hief_sigma2 = create_hief_dataset(filename=filename, output_filename=output_filename_2, noise_type=2)
    else:
        print('File ',output_filename_2, ' already exists')
        hief_sigma2 = pd.read_excel(output_filename_2).to_numpy()

    print ('Creating HIEF with noise type 3, smaller')
    if not path.exists(output_filename_3):
        hief_smaller = create_hief_dataset(filename=filename, output_filename=output_filename_3, noise_type=3)
    else:
        print('File ',output_filename_3, ' already exists')
        hief_smaller = pd.read_excel(output_filename_3).to_numpy()

    do_plots(hief_original, hief_sigma1, hief_sigma2, hief_smaller)
