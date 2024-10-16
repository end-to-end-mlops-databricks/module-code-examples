from src.data_processor import DataProcessor
from src.price_model import PriceModel
from src.utils import visualize_results, plot_feature_importance
import yaml

# Load configuration
with open('project_config.yml', 'r') as file:
    config = yaml.safe_load(file)

print("Configuration loaded:")
print(yaml.dump(config, default_flow_style=False))


# Initialize DataProcessor
data_processor = DataProcessor('data/data.csv', config)

# Preprocess the data
data_processor.preprocess_data()

# Split the data
X_train, X_test, y_train, y_test = data_processor.split_data()

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# Initialize and train the model
model = PriceModel(data_processor.preprocessor, config)
model.train(X_train, y_train)

# Evaluate the model
mse, r2 = model.evaluate(X_test, y_test)
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")


## Visualizing Results
y_pred = model.predict(X_test)
visualize_results(y_test, y_pred)

## Feature Importance
feature_importance, feature_names = model.get_feature_importance()
plot_feature_importance(feature_importance, feature_names)