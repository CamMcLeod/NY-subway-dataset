import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt

"""
In this optional exercise, you should complete the function called 
predictions(turnstile_weather). This function takes in our pandas 
turnstile weather dataframe, and returns a set of predicted ridership values,
based on the other information in the dataframe.  

You should attempt to implement another type of linear regression, 
that you may have read about, such as ordinary least squares regression: 
http://en.wikipedia.org/wiki/Ordinary_least_squares

This is your playground. Go wild!

How does your choice of linear regression compare to linear regression
with gradient descent?

You can look at the information contained in the turnstile_weather dataframe below:
https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv

Note: due to the memory and CPU limitation of our amazon EC2 instance, we will
give you a random subset (~15%) of the data contained in turnstile_data_master_with_weather.csv

If you receive a "server has encountered an error" message, that means you are hitting 
the 30 second limit that's placed on running your program. See if you can optimize your code so it
runs faster.
"""

def predictions(weather_turnstile):
    # Select Features (try different features!)
    features = weather_turnstile[['rain', 'hour', 'meantempi','weekday']]
    
    # Add UNIT to features using dummy variables
    dummy_units = pd.get_dummies(weather_turnstile[['UNIT', 'conds']], prefix=['unit', 'conds'])
    features = features.join(dummy_units)
    
    # Values
    values = weather_turnstile['ENTRIESn_hourly']
    

    features, mu, sigma = normalize_features(features)
    
    
    # Convert features and values to numpy arrays
    features_array = np.array(features)
    values_array = np.array(values)
    features_array = sm.tools.add_constant(features_array)
    
    # plug values and features into OLS model
    model = sm.OLS(values_array,features_array)
    results = model.fit()
    prediction = results.predict(features_array)
    print results.summary()
    return prediction

def compute_r_squared(data, predictions):
    SST = ((data-np.mean(data))**2).sum()
    SSReg = ((predictions-np.mean(data))**2).sum()
    r_squared = SSReg / SST

    return r_squared

def plot_residuals(turnstile_weather, predictions):
    '''
    Using the same methods that we used to plot a histogram of entries
    per hour for our data, why don't you make a histogram of the residuals
    (that is, the difference between the original hourly entry data and the predicted values).
    Try different binwidths for your histogram.

    Based on this residual histogram, do you have any insight into how our model
    performed?  Reading a bit on this webpage might be useful:

    http://www.itl.nist.gov/div898/handbook/pri/section2/pri24.htm
    '''
    
    plt.figure()
    plt.axis([-5000, 5000, 0, 4000])
    plt.legend(loc='upper right')
    plt.ylabel('Frequency')
    plt.title('Histogram of Residuals')
    plt.xlabel('ENTRIESn_hourly - predictions')
    (turnstile_weather['ENTRIESn_hourly'] - predictions).hist(bins = 150)
    return plt

def plot_residuals_diff(turnstile_weather, predictions):
    plt.figure()    
    my_plot = scipy.stats.probplot(turnstile_weather['ENTRIESn_hourly'] - predictions, plot=plt)
    plt.show()
    return my_plot

def normalize_features(array):
   """
   Normalize the features in our data set.
   """
   array_normalized = (array-array.mean())/array.std()
   mu = array.mean()
   sigma = array.std()

   return array_normalized, mu, sigma

if __name__ == "__main__":
    input_filename = "turnstile_weather_v2.csv"
    turnstile_master = pd.read_csv(input_filename)
    predicted_values = predictions(turnstile_master)
    r_squared = compute_r_squared(turnstile_master['ENTRIESn_hourly'], predicted_values) 
    
    image = "plot.png"
    plt = plot_residuals(turnstile_master, predicted_values)
    plt2 = plot_residuals_diff(turnstile_master, predicted_values)
    plt.savefig(image)
    
    print "R squared: %s" % r_squared