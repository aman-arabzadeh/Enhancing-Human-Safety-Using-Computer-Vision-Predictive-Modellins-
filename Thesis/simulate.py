import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
############################### Kalman Filtering ###################################
# Kalman Filtering
data_KF = pd.DataFrame({
    'timestamp_KF': [
        1714557212.6273396, 1714557212.7191963, 1714557212.8113885, 1714557212.90399,
        1714557213.0057094, 1714557213.1272337, 1714557213.3365836, 1714557213.4318686,
        1714557213.5234292, 1714557213.616757, 1714557213.7150733, 1714557213.808764,
        1714557213.8950133, 1714557213.9791713, 1714557214.0598528, 1714557214.1531427,
        1714557214.2428474, 1714557214.3313541, 1714557214.4248073, 1714557214.5150352,
        1714557214.6067255, 1714557214.697825, 1714557214.785114, 1714557214.873678,
        1714557214.9610372, 1714557215.0464957, 1714557215.1341994, 1714557215.2240388,
        1714557215.3063421, 1714557215.387342
    ],
    'det_x_KF': [
        1143, 1143, 1144, 1143, 1143, 1143, 1143, 1143, 1143, 1143, 1145, 1143,
        1144, 1145, 1144, 1144, 1143, 1143, 1141, 1141, 1140, 1140, 1140, 1140,
        1139, 1139, 1136, 1136, 1131, 1130
    ],
    'det_y_KF': [
        240, 240, 240, 240, 239, 239, 237, 237, 235, 235, 234, 234, 232, 232,
        229, 229, 225, 225, 223, 223, 220, 220, 216, 216, 212, 212, 208, 208,
        200, 200
    ],
    'pred_x_KF': [
        1143, 1143, 1143, 1144, 1143, 1142, 1142, 1142, 1142, 1142, 1142, 1145,
        1143, 1143, 1145, 1144, 1143, 1142, 1142, 1140, 1140, 1139, 1139, 1139,
        1139, 1138, 1138, 1135, 1134, 1129
    ],
    'pred_y_KF': [
        240, 240, 240, 240, 240, 238, 238, 236, 236, 234, 234, 233, 233, 231,
        231, 227, 227, 223, 223, 221, 221, 218, 218, 214, 214, 209, 210, 205,
        206, 196
    ],
    'class_name_KF': ['frisbee'] * 30  # Assuming all entries are 'frisbee'
})

# Calculate MSE and MAE for the x coordinates
mse_x_KF = mean_squared_error(data_KF['det_x_KF'], data_KF['pred_x_KF'])
mae_x_KF = mean_absolute_error(data_KF['det_x_KF'], data_KF['pred_x_KF'])

# Calculate MSE and MAE for the y coordinates
mse_y_KF = mean_squared_error(data_KF['det_y_KF'], data_KF['pred_y_KF'])
mae_y_KF = mean_absolute_error(data_KF['det_y_KF'], data_KF['pred_y_KF'])

print("Kalman Filtering dynamic video movemnets sample")
print("MSE X:", mse_x_KF, "MAE X:", mae_x_KF)
print("MSE Y:", mse_y_KF, "MAE Y:", mae_y_KF)
print("______________________________________________________________")


############################### Dead Reckoning ###################################
# Dead Reckoning Data
data_DR = {
    'timestamp_DR': [
        1714568442.4965265, 1714568442.578254, 1714568442.6561844, 1714568442.7386522, 1714568442.8225925,
        1714568442.907595, 1714568443.2264352, 1714568443.3127913, 1714568443.3956892, 1714568443.4956355,
        1714568443.596996, 1714568443.693001, 1714568443.7796285, 1714568443.872993, 1714568443.9699917,
        1714568444.0675254, 1714568444.1653264, 1714568444.2561796, 1714568444.3441617, 1714568444.4352129,
        1714568444.527983, 1714568444.6203768, 1714568444.7060127, 1714568444.7865372, 1714568444.869198,
        1714568444.9558237, 1714568445.0370693, 1714568445.115068, 1714568445.212649, 1714568445.3590412
    ],
    'det_x_DR': [
        1143, 1143, 1144, 1143, 1143, 1143, 1143, 1143, 1143, 1143, 1145, 1143,
        1144, 1145, 1144, 1144, 1143, 1143, 1141, 1141, 1140, 1140, 1140, 1140,
        1139, 1139, 1136, 1136, 1131, 1130
    ],
    'det_y_DR': [
        240, 240, 240, 240, 239, 239, 237, 237, 235, 235, 234, 234, 232, 232,
        229, 229, 225, 225, 223, 223, 220, 220, 216, 216, 212, 212, 208, 208,
        200, 200
    ],
    'pred_x_DR': [
        1143, 1143, 1145, 1142, 1143, 1143, 1143, 1143, 1143, 1143, 1147, 1141,
        1145, 1146, 1143, 1144, 1142, 1143, 1139, 1141, 1139, 1140, 1140, 1140,
        1138, 1139, 1133, 1136, 1126, 1129
    ],
    'pred_y_DR': [
        240, 240, 240, 240, 238, 239, 235, 237, 233, 235, 233, 234, 230, 232,
        226, 229, 221, 225, 221, 223, 217, 220, 212, 216, 216, 212, 204, 208,
        192, 200
    ],
    'class_name': ['frisbee'] * 30  # Assuming all entries are 'frisbee'
}

# Calculate MSE and MAE for the x coordinates
mse_x_DR = mean_squared_error(data_DR['det_x_DR'], data_DR['pred_x_DR'])
mae_x_DR = mean_absolute_error(data_DR['det_x_DR'], data_DR['pred_x_DR'])

# Calculate MSE and MAE for the y coordinates
mse_y_DR = mean_squared_error(data_DR['det_y_DR'], data_DR['pred_y_DR'])
mae_y_DR = mean_absolute_error(data_DR['det_y_DR'], data_DR['pred_y_DR'])

print("Dead Reckoning dynamic video movemnets sample")
print("MSE X DR:", mse_x_DR, "MAE X DR:", mae_x_DR)
print("MSE Y DR:", mse_y_DR, "MAE Y DR:", mae_y_DR)

print("______________________________________________________________")

############################## Kalman Filter Linear ###################################
dataLinear = {
    "timestamp_KF_Linear": [
        1714577070.6049674, 1714577070.6990438, 1714577070.8075929, 1714577071.4000566,
        1714577071.4718778, 1714577071.5430427, 1714577071.615306, 1714577071.674309,
        1714577071.7473059, 1714577071.8179853, 1714577071.886072, 1714577071.9500747,
        1714577072.0230765, 1714577072.0900722, 1714577072.1560724, 1714577072.227071,
        1714577072.293354, 1714577072.3638673, 1714577072.3638673, 1714577072.4308143,
        1714577072.4318168, 1714577072.5049891, 1714577072.5049891, 1714577072.5645742,
        1714577072.633891, 1714577072.7047405, 1714577072.7755148, 1714577072.8409805,
        1714577072.9082556, 1714577072.9830308
    ],
    "det_x_KF_Linear": [
        1106, 1106, 1106, 1067, 1067, 1040, 1040, 1040, 1025, 1025, 1025, 1025,
        1025, 1000, 1000, 1000, 1000, 973, 973, 973, 973, 973, 973, 959, 959,
        946, 946, 946, 933, 933
    ],
    "det_y_KF_Linear": [
        158, 158, 158, 197, 197, 225, 225, 225, 238, 238, 238, 238, 238, 264, 264,
        264, 264, 291, 292, 291, 292, 291, 292, 304, 304, 318, 318, 318, 331, 331
    ],
    "pred_x_KF_Linear": [
        1106, 1106, 1106, 1067, 1067, 1106, 1024, 1016, 1024, 1014, 1016, 1020,
        1023, 1067, 986, 976, 984, 992, 1024, 963, 960, 962, 954, 961, 949, 951,
        938, 939, 966, 920
    ],
    "pred_y_KF_Linear": [
        158, 158, 158, 197, 197, 158, 240, 249, 240, 248, 245, 241, 239, 197, 277,
        287, 279, 271, 238, 300, 305, 301, 311, 304, 312, 310, 324, 324, 297, 343
    ],
    "class_name_KF_Linear": [
        "sports ball", "sports ball", "sports ball", "frisbee", "frisbee", "sports ball",
        "sports ball", "sports ball", "sports ball", "sports ball", "sports ball",
        "sports ball", "sports ball", "frisbee", "frisbee", "frisbee", "frisbee",
        "frisbee", "sports ball", "frisbee", "sports ball", "frisbee", "sports ball",
        "sports ball", "sports ball", "sports ball", "sports ball", "sports ball",
        "frisbee", "frisbee"
    ]
}

# Create DataFrame
data_KF_Linear = pd.DataFrame(dataLinear)
# Calculate MSE and MAE for the x coordinates
mse_x_KF_Linear = mean_squared_error(data_KF_Linear['det_x_KF_Linear'], data_KF_Linear['pred_x_KF_Linear'])
mae_x_KF_Linear = mean_absolute_error(data_KF_Linear['det_x_KF_Linear'], data_KF_Linear['pred_x_KF_Linear'])

# Calculate MSE and MAE for the y coordinates
mse_y_KF_Linear = mean_squared_error(data_KF_Linear['det_y_KF_Linear'], data_KF_Linear['pred_y_KF_Linear'])
mae_y_KF_Linear = mean_absolute_error(data_KF_Linear['det_y_KF_Linear'], data_KF_Linear['pred_y_KF_Linear'])
print()

print("Kalman Filtering Linear video movemnets sample")

print("MSE X:", mse_x_KF_Linear, "MAE X:", mae_x_KF_Linear)
print("MSE Y:", mse_y_KF_Linear, "MAE Y:", mae_y_KF_Linear)
print("______________________________________________________________")


############################# Dead Reckoning Linear ####################################


data_DR_Linears = {
    "timestamp": [
        1714577098.6916099, 1714577098.7584355, 1714577098.8345513, 1714577099.474594,
        1714577099.540503, 1714577099.6121929, 1714577099.688144, 1714577099.7531729,
        1714577099.8219564, 1714577099.88716, 1714577099.9501245, 1714577100.0146544,
        1714577100.0814316, 1714577100.1523466, 1714577100.2181249, 1714577100.2855935,
        1714577100.348358, 1714577100.418886, 1714577100.4198885, 1714577100.4856005,
        1714577100.4856005, 1714577100.5547602, 1714577100.5547602, 1714577100.6199737,
        1714577100.6914515, 1714577100.7604318, 1714577100.829523, 1714577100.8973525,
        1714577100.9570947, 1714577101.022565
    ],
    "det_x_DR_Linear": [
        1106, 1106, 1106, 1067, 1067, 1040, 1040, 1040, 1025, 1025, 1025, 1025,
        1025, 1000, 1000, 1000, 1000, 973, 973, 973, 973, 973, 973, 959, 959,
        946, 946, 946, 933, 933
    ],
    "det_y_DR_Linear": [
        158, 158, 158, 197, 197, 225, 225, 225, 238, 238, 238, 238, 238, 264, 264,
        264, 264, 291, 292, 291, 292, 291, 292, 304, 304, 318, 318, 318, 331, 331
    ],
    "pred_x_DR_Linear": [
        1106, 1106, 1106, 1067, 1067, 974, 1040, 1040, 1010, 1025, 1025, 1025,
        1025, 933, 1000, 1000, 1000, 946, 921, 973, 973, 973, 973, 945, 959,
        933, 946, 946, 893, 933
    ],
    "pred_y_DR_Linear": [
        158, 158, 158, 197, 197, 292, 225, 225, 251, 238, 238, 238, 238, 331, 264,
        264, 264, 318, 346, 291, 292, 291, 292, 316, 304, 332, 318, 318, 371, 331
    ],
    "class_name_Linear": [
        "sports ball", "sports ball", "sports ball", "frisbee", "frisbee", "sports ball",
        "sports ball", "sports ball", "sports ball", "sports ball", "sports ball",
        "sports ball", "sports ball", "frisbee", "frisbee", "frisbee", "frisbee",
        "frisbee", "sports ball", "frisbee", "sports ball", "frisbee", "sports ball",
        "sports ball", "sports ball", "sports ball", "sports ball", "sports ball",
        "frisbee", "frisbee"
    ]
}

# Create DataFrame
data_DR_Linear = pd.DataFrame(data_DR_Linears)
# Calculate MSE and MAE for the x coordinates
mse_x_DR_Linear = mean_squared_error(data_DR_Linear['det_x_DR_Linear'], data_DR_Linear['pred_x_DR_Linear'])
mae_x_DR_Linear = mean_absolute_error(data_DR_Linear['det_x_DR_Linear'], data_DR_Linear['pred_x_DR_Linear'])

# Calculate MSE and MAE for the y coordinates
mse_y_DR_Linear = mean_squared_error(data_DR_Linear['det_y_DR_Linear'], data_DR_Linear['pred_y_DR_Linear'])
mae_y_DR_Linear = mean_absolute_error(data_DR_Linear['det_y_DR_Linear'], data_DR_Linear['pred_y_DR_Linear'])

print("Dead Reckoning  Linear video movemnets sample")
print("MSE X DR:", mse_x_DR_Linear, "MAE X DR:", mae_x_DR_Linear)
print("MSE Y DR:", mse_y_DR_Linear, "MAE Y DR:", mae_y_DR_Linear)
