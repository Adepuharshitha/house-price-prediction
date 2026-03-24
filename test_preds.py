import joblib
import numpy as np

model = joblib.load('house_model.pkl')

pred1 = model.predict(np.array([[2000, 3, 2, 1, 1]]))
pred2 = model.predict(np.array([[5000, 4, 3, 2, 2]]))
pred3 = model.predict(np.array([[3000, 2, 1, 1, 0]]))

print("Prediction 1:", pred1)
print("Prediction 2:", pred2)
print("Prediction 3:", pred3)
