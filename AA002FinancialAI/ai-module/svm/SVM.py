import numpy as np

class SVM:

    def __init__(self, learning_rate = 0.0001, epochs = 1, lambda_param = 0.05):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_param = lambda_param
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        Train the optimal 3D SVM hyperplane using Stochastic Gradient Descent (SGD).

        Overall Objective Function to minimize:
            J(w, b) = lambda * ||w||^2 + sum( Hinge_Loss(x_i, y_i) )

        Where:
        - Component 1 (L2 Regularization Term): lambda * ||w||^2
            Controls the smoothness of the hyperplane and decays the weight vector (w) 
            to maximize the margin, preventing systemic overfitting.
        - Component 2 (Classification Penalty / Hinge Loss): max(0, 1 - y_i * (w.x_i + b))
            Acts as a mathematical switch triggered when a data point is misclassified 
            or falls inside the dangerous margin zone.

        Weight Update Mechanism (Gradient Descent):
        - If y_i * (w.x_i + b) < 1 (Margin Violation / Misclassification):
            w = w - lr * (2 * lambda * w - y_i * x_i)  -> Weight decay combined with error correction.
        - Else (Safe Zone outside the margin):
            w = w - lr * (2 * lambda * w)              -> Pure Regularization (Weight decay only).
        """
        X_mat = np.array(X)
        y_vec = np.array(y)
        n_samples, n_features = X_mat.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for epoch in range(self.epochs):
            
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for idx in indices:
                
                x_i = X_mat[idx]
                y_i = y_vec[idx]
                
                linear_model = np.dot(x_i, self.w) + self.b
                # Hinge Loss condition check: y_i * (w.x + b) < 1
                if y_i * linear_model < 1:
                    # CASE A: Misclassification OR Margin Violation (Dangerous Zone)
                    # Update weights by combining Error Correction and Weight Decay (L2 Regularization)
                    # - '2 * self.lambda_param * self.w': Pushes weights toward 0 to maximize the margin
                    # - '- y_i * x_i': Bends the hyperplane to correctly classify this specific data point
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - y_i * x_i)
                    self.b += self.learning_rate * y_i
                else:
                    # CASE B: Correctly classified AND safely outside the margin (Safe Zone)
                    # Hinge Loss is exactly 0. No classification error to correct.
                    # Run pure L2 Regularization (Weight Decay) to shrink 'w' and maximize the margin width
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)


    def predict(self, X):
        X_mat = np.array(X)
        approx = X_mat @ self.w + self.b
        return np.sign(approx)
    
    def score(self, X, y):
        predictions =  self.predict(X)
        return np.mean(predictions == y)
