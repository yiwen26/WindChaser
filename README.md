## Data pre-processing

### Abstruct

The purpose of oue data pre-processing is to overview roughly with our data. We resampled our data and calculate distibution and correlation coefficient, meanwhile visualized each data from diverse aspects.


### Dataset we use

* National Renewable Energy Laboratory (NREL) Wind Integration Datasets
* Location: Seattler
* Year: 2012
* Data: power (MW); 
        wind speed at 100m (m/s); 
        wind direction at 100m (deg); 
        air temperature at 2m (K); 
        surface air pressure (Pa); 
        density at hub height (kg/m^3)


### Instructions 

More details are in the Jupyter notebook named `Wind_data_pre-processing`. Follow through the notebook and you can see features of our wind data.
The work we do in this part is try to understand the data quickly. We made efforts to construct some traditional simple model to analyze the power. Among all the kinds of data, you can only found that wind speed is more related with power generation (their correlation coefficient is very close to 1.0). So we designed a strategy that we can use the natural data 'wind speed' to make prediction of power. But you can also see that the auto-correlation coefficient of speed and power is much far away form 1.0, it is around 0! Hence we choose to construct a re-current neural networks to forecasting the power.



