import joblib
import pandas as pd

try:
    feature_cols = joblib.load('feature_cols.pkl')
    scaler = joblib.load('scaler_mantenimiento_predictivo.pkl')
    model = joblib.load('modelo_mantenimiento_predictivo.pkl')
    
    print("Feature Columns:", feature_cols)
    
    # Case 1
    case1_dict = {
        'metric1': 215630672.0,
        'metric2': 55.0,
        'metric4': 52.0,
        'metric5': 6.0,
        'metric6': 407438.0,
        'metric7': 0.0,
        'metric9': 7.0
    }
    case1_df = pd.DataFrame([[case1_dict[col] for col in feature_cols]], columns=feature_cols)
    case1_scaled = pd.DataFrame(scaler.transform(case1_df), columns=feature_cols)
    case1_prob = model.predict_proba(case1_scaled)[0, 1]
    print("Case 1 probability:", case1_prob)
    
    # Case 2
    case2_dict = {
        'metric1': 0.0,
        'metric2': 0.0,
        'metric4': 0.0,
        'metric5': 0.0,
        'metric6': 0.0,
        'metric7': 0.0,
        'metric9': 0.0
    }
    case2_df = pd.DataFrame([[case2_dict[col] for col in feature_cols]], columns=feature_cols)
    case2_scaled = pd.DataFrame(scaler.transform(case2_df), columns=feature_cols)
    case2_prob = model.predict_proba(case2_scaled)[0, 1]
    print("Case 2 probability:", case2_prob)
    
    print("Verification completed successfully!")
except Exception as e:
    print("Verification failed:", e)
    import traceback
    traceback.print_exc()
