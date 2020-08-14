# Final Project For AIHUB (Magnificent G1)
## This project aims to make estimation about the COVID-19 situation in the US

### Our Thoughts
Predicting the impact of the COVID-19, including daily increases, confirmed positives and etc. is critically significant on how to faithfully represents the real epidemic situation, and therefore, guiding us to handle with it.
We implement several estimation approach to make "precise" (not always) predictions, including **Lasso Regression, Ridge Regression, Multivariate Polynomial Regression, FCNN (Fully-Connected Nueral Network) and LSTM**. Also, we build an **classification model based on twitter sentimenal analysis**, using overall percentage of different emotion types to predict a plausible  range of daily increase number. We also combine **SEIR model with the nueral network,** utilizing the NN to sovle ODE problem.

All the codes relate to data are saved in *"dataset"* folder. 

### Regression Models
All the codes relate to regression are saved in *"regression"* folder.
To fit the daily increase positive cases number, we split the whole range during 15th Mar to 7th Aug in to 3 time spans. The first one ranges from 15th Mar to 15th Apr, the second ranges from 16th Apr to 15th Jul, and the third one ranges from 16th Jul yo 7th Aug. This spilt maximizes the peak feature of each span.

Lasso Regression, Ridge Regression, Multivariate Polynomial Regression are used to fit data on spans aforementioned. We use MSE to select the best fit model for each span. In cosideration of the relativity of consecutive days, we design a **time window function** to load data in a specific range with different adaptive weights for each day in this range.This processing helps models to fit data better since in real world, data of several consecutive days are releated.

We try to fit specific state data to find if selected states can generally represent the whole US epidemic trend with linear regression, and the result is amazing. Data of GA and MA can almost perfectly fit the whole US data.

### Nueral Network Models
**Important**: Due to the deciciency of reliable data, NN models perform not very well. But we believe our models are potentialy promising.

All the codes relate to nueral netowk are saved in *"neural netowrk"* folder.

FCNN is the simplest NN model, we use it as a "hand-on experiment".

LSTM **was** a "state-of-the-art" model to predict time series data. The most important feature of it is that it combines previous output with current input to feed the neurons. So, LSTM includes a time window in it. This addresses either gradient vanish or gradient explode in long series training.

We done something really fun in our project, which is the classification model based on twitter sentimenal analysis. We collect some twitters in the US from 10th Jul to 7th Aug and analyze those twitters' emotional attitude towards COVID-19 (positive, negative or nuetral). And the responsible variable is daily increase positive cases. As we input the ratio of those three attitudes on a specific day, the model can predict the increase cases of this day. 
(55000-  -->1    55000-60000  -->2    60000-65000  -->3   65000-70000  -->4    70000+  -->5)

The most creative and hardcore portion of our project is SEIR model with the nueral network. Considering the fundamental goal of ODE is to calculate equations like xxxxx = 0, it is perfect to work with neural network whose goal is to minimize the loss to 0. We "transform" SEIR equations in to loss functions in a neural network, trying to fit the analytical solution of those equations.
