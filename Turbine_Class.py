import pandas as pd
import matplotlib.pyplot as plt
import logging
from scipy.stats import pearsonr
import numpy as np
import glob

logging.basicConfig(level=logging.INFO, filename = 'info.log',  filemode = 'w')

class Turbine_data:
    def __init__(self, base, turbine_number, year, output_file = ['Wind speed (m/s)', 'Wind direction (°)', 'Nacelle position (°)', 'Vane position 1+2 (°)', 'Energy Export (kWh)', 'Energy Import (kWh)', 
                            'Power (kW)', 'Energy Theoretical (kWh)', 'Lost Production to Downtime (kWh)', 'Lost Production to Performance (kWh)',  
                            'Lost Production Total (kWh)', 'Generator RPM (RPM)', 'Generator bearing rear temperature (°C)', 'Rotor bearing temp (°C)', 'Ambient temperature (converter) (°C)']):
        # load and transform the data into a workable format
        
        # base (str): directory where the data is stored
        # turbine number (str): number of the turbine investigated
        # year (str): year of investigation
        # output_file (list of str): list of column names to extract
        
        files = glob.glob(base+'ONYX_data/Turbine{}/Turbine_Data_Penmanshiel_{}_{}*.csv'.format(turbine_number, turbine_number, year))

        if not files:
            raise FileNotFoundError("No data files found for Turbine {}, year {}".format(turbine_number,year))

        Turbine_data = pd.read_csv(files[0], skiprows=9,  low_memory=False, index_col=False)
        
        #change to datetime object
        Turbine_data.columns = ['Timestamp'] + list(Turbine_data.columns[1:])
        Turbine_data['Timestamp'] = pd.to_datetime(Turbine_data['Timestamp'])
        Turbine_data.set_index('Timestamp', inplace=True)
        
        #drop columns where every value is NaN
        dropped_nans = Turbine_data.dropna(axis=1, how='all')
        print(dropped_nans.columns.tolist())
        
        
        #sanity checks
        if (Turbine_data['Wind direction (°)'] < 0).any() or  (Turbine_data['Wind direction (°)'] > 360).any():
            logging.debug('error with wind direction!')
        if (Turbine_data['Nacelle position (°)'] < 0).any() or  (Turbine_data['Nacelle position (°)'] > 360).any():
            logging.debug('error with Nacelle position!')
        if (Turbine_data['Energy Export (kWh)'] < -1000).any():
            logging.debug('error with energy export!')
        
        #read in the status data
        files = glob.glob(base+'ONYX_data/Turbine{}/Status_Penmanshiel_{}_{}*.csv'.format(turbine_number,turbine_number,year))

        if not files:
            raise FileNotFoundError("No status files found for Turbine {}, year {}".format(turbine_number, year))
        status_data = pd.read_csv(files[0], comment='#').drop_duplicates()
        status_data['Timestamp start'] = pd.to_datetime(status_data['Timestamp start'])
        status_data['Timestamp end'] = status_data['Timestamp end'].mask(status_data['Timestamp end'] == '-',status_data['Timestamp start'])
        status_data['Timestamp end'] = pd.to_datetime(status_data['Timestamp end'])    
        status_data['Duration'] = status_data['Duration'].mask(status_data['Duration'] == '-',0.0)
        status_data['Duration'] = pd.to_timedelta(status_data['Duration']).dt.total_seconds() / 3600
        
        #define objects in class 
        #extract just relavent column names
        self.turbine_data = Turbine_data[output_file].copy()
        self.status_data = status_data
        self.base = base
        self.T_number = turbine_number
        self.year = year
    
    
    def select_unavailiable_timeperiods(self, thresh=5.0):
        #find unavaiable time periods where the power remains low for a high wind speed
        
        # thresh (float): define high windspeed threshold in m/s
    
        # Select the unaviable modes
        mask = (self.turbine_data['Wind speed (m/s)'] > thresh) & (self.turbine_data['Power (kW)'] < 0)
        indexes = self.turbine_data.loc[mask].index

        logging.info("Showing the dataframe of unavaiable times \n")
        logging.info(self.turbine_data.loc[indexes, ['Wind speed (m/s)', 'Power (kW)']])

        #plot with no unavaiable times
        data_avaiable = self.turbine_data.drop(indexes)

        plt.scatter(self.turbine_data['Wind speed (m/s)'], self.turbine_data['Power (kW)'], marker='x', label='T{}_{}'.format(self.T_number, self.year))
        plt.scatter(data_avaiable['Wind speed (m/s)'], data_avaiable['Power (kW)'], marker='x', label='T{}_{}_avail'.format(self.T_number, self.year))
        plt.xlabel(r'$ \rm Wind Speed /ms^{-1}$')
        plt.ylabel(r'$ \rm Power /kW$')
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(self.base+'/plots/avaiable_power_curve_T{}_{}.pdf'.format(self.T_number, self.year))
        plt.show()
        
        self.unavaiable_idxs = indexes
    
    def compare_downtime_reasons(self, lost_production_columns=['Lost Production to Downtime (kWh)', 'Lost Production to Performance (kWh)']):
        #compare downtime epochs to power to see the reasons behind the biggest drops in power
        
        #lost_production_columns (list of str): list of columns with which to compare downtime
        
        fig, ax = plt.subplots(2, figsize=(14,8), sharex=True)

        ax[0].plot(self.turbine_data.index, (self.turbine_data['Energy Export (kWh)']-self.turbine_data['Energy Import (kWh)'])/self.turbine_data['Energy Theoretical (kWh)'], label='Net energy')
        ax[0].axhline(y=1,linestyle='dashed', color='k')
        ax[0].set_yscale('log')
        for ii in lost_production_columns:
            ax[1].plot(self.turbine_data.index, self.turbine_data[ii], alpha = 0.5, label=ii)
        ax[1].legend(frameon=False)
        ax[0].set_ylabel('True/Therorectial Output')
        ax[1].set_ylabel('Energy /kWh')
        ax[1].set_xlabel('Timestamp')
        
        for a in ax:
            for idx in self.unavaiable_idxs :
                a.axvline(x=idx, color='red', alpha=0.05, linewidth=0.5)
        plt.tight_layout()
        plt.savefig(self.base+'/plots/downtime_timeseries_T{}_{}.pdf'.format(self.T_number, self.year))
        plt.show()
        
        indexes_downtime = self.turbine_data.loc[self.turbine_data['Lost Production to Downtime (kWh)']>0].index
        indexes_performance = self.turbine_data.loc[self.turbine_data['Lost Production to Performance (kWh)']>0].index

        fig, ax = plt.subplots(2, figsize=(14,8), sharex=True)
        ax[0].set_title('Comparing Lost Energy Output Reasons')
        ax[0].plot(self.turbine_data.index, (self.turbine_data['Energy Export (kWh)']-self.turbine_data['Energy Import (kWh)'])/self.turbine_data['Energy Theoretical (kWh)'] , label='Net energy')
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')
        ax[1].axhline(y=1,linestyle='dashed', color='k')
        ax[1].axhline(y=1,linestyle='dashed', color='k')
        ax[1].plot(self.turbine_data.index, (self.turbine_data['Energy Export (kWh)']-self.turbine_data['Energy Import (kWh)'])/self.turbine_data['Energy Theoretical (kWh)'], label='Net energy')
        for idx in indexes_downtime:
                ax[0].axvline(x=idx, color='red', alpha=0.05, linewidth=0.5)
        for idx in indexes_performance:
                ax[1].axvline(x=idx, color='green', alpha=0.05, linewidth=0.5)
        plt.tight_layout()
        plt.savefig(self.base+'/plots/downtime_reasons_T{}_{}.pdf'.format(self.T_number, self.year))
        plt.show()
        
    def compare_timeseries_data(self, column_names_to_compare, optimum_values=None):
        #compare different timeseries datasets and output the correlation cofficient
        
        #column_names_to_compare (list of str): column names of which to compare
        #optimum_values (list of floats) OPTINAL: if there is an optimum value for each timeseries data provide a list of floats with which to add
        #                                         comparison lines on the figure
        
        fig, axs = plt.subplots(len(column_names_to_compare), sharex=True)
        filename = 'plots/comparison_'
        for ii, name in enumerate(column_names_to_compare):
            self.turbine_data[name].resample('D').mean().plot(
            ylabel=name,
            ax=axs[ii])
            if optimum_values is not None:
                axs[ii].axhline(y=optimum_values[ii], color='k', linestyle='dashed', alpha=0.05, linewidth=0.5)
            safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_per_')
            filename += safe_name + '_'
        # Add unavailable periods as vertical lines
        for ax in axs:
            for idx in self.unavaiable_idxs :
                ax.axvline(x=idx, color='red', alpha=0.05, linewidth=0.5)
        plt.tight_layout()
        plt.savefig(self.base+filename+'T{}_{}.pdf'.format(self.T_number, self.year))
        plt.show()
        
        
        if len(column_names_to_compare) == 2:
            self.turbine_data[column_names_to_compare[0]] = self.turbine_data[column_names_to_compare[0]].replace([np.nan, -np.inf], 0)
            self.turbine_data[column_names_to_compare[1]] = self.turbine_data[column_names_to_compare[1]].replace([np.nan, -np.inf], 0)
            corr, p_value = pearsonr(self.turbine_data[column_names_to_compare[0]], self.turbine_data[column_names_to_compare[1]])
            print(f"PCC: {corr:.3f}, p-value: {p_value:.3f}")

    def find_downtime_reason(self, date_index):
        #Given a specific index of a list of datetime objects where the turbine is unavaiable then provide the reason for the downtime
        
        #date_index (int): index of interest
        
        date = self.unavaiable_idxs[date_index]
        
        mask = (self.status_data['Timestamp start'] <= date) & \
           (self.status_data['Timestamp end'] >= date)
        matches = self.status_data[mask]
        
        if matches.empty:
            print(f"No downtime found for {date}")
        else:
            for ii in range(0, len(matches)):
                print(ii)
                print('Downtime for {:.1f} hours on {} due to {} \n'.format(matches['Duration'].iloc[ii], date, matches['Message'].iloc[ii]))

            
    def temp_comparison(self, temperature_names):
        #plot temperature vs power for various temperature data, where the temperature is normalised to the amient temperature
        
        #temperature_names (list of str): A list of the different temperature column name which to compare
        
        filename = 'plots/comparison_'
        
        fig, axs = plt.subplots(len(temperature_names), sharex=True)
        for ii, name in enumerate(temperature_names):
            axs[ii].scatter(self.turbine_data[name]-self.turbine_data['Ambient temperature (converter) (°C)'], self.turbine_data['Power (kW)'], marker='x')
            axs[ii].set_xlabel(name)
            axs[ii].set_ylabel(r'$ \rm Power /kW$')
            #for idx in self.unaviable_indexes:
            #    axs[ii].axvline(x=idx, color='red', alpha=0.05, linewidth=0.5)
            safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_per_')
            filename += safe_name + '_'
            
        plt.tight_layout()
        plt.savefig(self.base+filename+'T{}_{}.pdf'.format(self.T_number, self.year))
        plt.show()
        