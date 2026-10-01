from ml_models import train_damage_model, train_risk_model
print("Training image damage classifier...")
train_damage_model()
print("Training structural risk model...")
train_risk_model()
print("Models saved in models/")
