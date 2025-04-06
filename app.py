from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import pandas as pd
import numpy as np
import difflib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
file_path = "mobile.csv"
df = pd.read_csv(file_path)

# Store names separately before dropping
mobile_names = df["Name"].tolist()
lowered_mobile_names = [name.strip().lower() for name in mobile_names]
df = df.drop(columns=["Unnamed: 0", "Model"])  # Keep "Name"

# Encode categorical variables
categorical_cols = ["Brand", "Operating system", "Touchscreen", "Wi-Fi", "Bluetooth", "GPS", "3G", "4G/ LTE"]
encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# Standardize numerical values
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.drop(columns=["Name"]))  # Exclude "Name" from scaling

# Compute cosine similarity
similarity_matrix = cosine_similarity(df_scaled)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/recommend', methods=['GET'])
def recommend():
    mobile_name = request.args.get('name', None)
    top_n = int(request.args.get('top_n', 5))
    
    if not mobile_name:
        return jsonify({"error": "No mobile name provided"}), 400
    
    mobile_name = mobile_name.strip().lower()
    
    # Find closest match
    matches = difflib.get_close_matches(mobile_name, lowered_mobile_names, n=1, cutoff=0.6)
    
    if not matches:
        return jsonify({"error": "Mobile not found"}), 404
    
    matched_name = matches[0]
    mobile_index = lowered_mobile_names.index(matched_name)
    
    # Get similarity scores
    similar_mobiles = list(enumerate(similarity_matrix[mobile_index]))
    similar_mobiles = sorted(similar_mobiles, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    # Extract mobile indices and names
    recommended_mobiles = [
        {"Name": mobile_names[idx], **df.iloc[idx].to_dict()}  # Include the mobile name in response
        for idx, _ in similar_mobiles
    ]
    
    return jsonify({
        "searched_mobile": mobile_names[mobile_index],  # Include the searched mobile name
        "recommendations": recommended_mobiles
    })

if __name__ == '__main__':
    app.run(debug=True)
