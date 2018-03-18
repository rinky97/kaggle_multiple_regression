# kaggle_multiple_regression

health insurance cost prediction


Problem: To predict the health insurance of different people basing on various factors like age, sex, number of children etc. 
Formalism: 
         Task(T) : creating a regression model basing on the dependence of health insurance on the various details of an individual.
         Experience(E): The training data that the model is fit to.
         Perfomance(P): To see how closely the predicted value of health insurance is to the actual value.
Assumptions: All the features help in determining the charges.
             So initially every feature must be preprocessed in a way that it would fit in the model.

Motivation: This is a real life problem which helps to approxiate how much an individual is granted insurance depending on his life style.

How to solve the problem:
                      Since there are multiple features, one has to decide which features impact the charges more than the others. So by using the method of backward elimination, we decide which features should remain in the model by comaparing the p values with the significance level.
                    if           p > SL  :then remove that feature
                    else   if    p < SL  :the feature remains
                    
 We use the statsmodel.formula.api library to create an object OLS: Ordinary Least Squares that gives us the various P values for each feature on using the summary method.
