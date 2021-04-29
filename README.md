# LightGBM_Regression
This repo shows how to use LightGBM for regression. A grid search is applied to improve the result which is then evaluate by means of the shap values.

![image](https://user-images.githubusercontent.com/71548024/116604775-dad64f80-a92e-11eb-998b-66fee1da6238.png)

### LightGBM Model

```
# Create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

print('Starting training...')
# Train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=200,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)

print('Starting predicting...')
# Predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)


# Plot
pyplot.plot(y_test, label='Expected')
pyplot.plot(y_pred, label='Predicted')
pyplot.legend()
pyplot.show()

# Evaluate
print("---------------------------------------------------------------")
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred))
print('The R2 score of the prediction is:', r2_score(y_test, y_pred))
print("---------------------------------------------------------------")
```
<img width="472" alt="Screenshot 2021-04-29 at 21 02 38" src="https://user-images.githubusercontent.com/71548024/116604356-4bc93780-a92e-11eb-9f01-080352e630b8.png">

### Shap Values
```
# Strat timer and instantiate the tree explainer
%time shap_values = shap.TreeExplainer(gbm).shap_values(X_test)
# Plot shap values (i.e. feature impact)
shap.summary_plot(shap_values, X_test)
```

<img width="465" alt="Screenshot 2021-04-29 at 21 02 48" src="https://user-images.githubusercontent.com/71548024/116604555-92b72d00-a92e-11eb-8283-9682137827ed.png">

