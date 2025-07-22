import pandas as pd

def predict_row(model, input_row, label_encoders=None, return_proba=False):
    """
    Predict a single row using a trained model.

    Args:
        model: Trained machine learning model with `.predict()` and optionally `.predict_proba()`.
        input_row (dict): Dictionary of feature values for a single input.
        label_encoders (dict, optional): Dictionary of LabelEncoders for categorical columns, if used during training.
        return_proba (bool): Whether to return prediction probabilities if supported.

    Returns:
        prediction: Predicted class or probability dictionary.
    """
    try:
        df = pd.DataFrame([input_row])

        # Apply label encoding if encoders provided
        if label_encoders:
            for col, encoder in label_encoders.items():
                if col in df.columns:
                    df[col] = encoder.transform(df[col].astype(str))

        prediction = model.predict(df)[0]

        if return_proba and hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)[0]
            return {"prediction": prediction, "probabilities": dict(enumerate(proba))}
        else:
            return prediction

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}
